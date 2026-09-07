"""Prepare a lightweight JSON index for the official BigEarthNet v2.0 S1/S2 layout.

The official archive stores each band as a separate GeoTIFF and keeps patch labels,
split, and S1<->S2 pairing in metadata.parquet. This script does NOT copy imagery;
it only creates train/val/test JSON index files consumed by BigEarthNetMMDataset.

Example:
  python scripts/prepare_bigearthnet_mm.py \
    --s2-root /content/BigEarthNet-S2 \
    --s1-root /content/BigEarthNet-S1 \
    --metadata /content/metadata.parquet \
    --out data/raw/bigearthnet_mm

Use --limit-per-split for a fast first run.
"""
import argparse, json, os
from pathlib import Path
import pandas as pd

S2_BANDS = ["B02", "B03", "B04", "B08"]
S1_BANDS = ["VV", "VH"]


def index_patch_dirs(root: Path, marker: str):
    mapping = {}
    for p in root.rglob(f"*_{marker}.tif"):
        patch_dir = p.parent
        mapping[patch_dir.name] = patch_dir
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-root", required=True)
    ap.add_argument("--s1-root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", default="data/raw/bigearthnet_mm")
    ap.add_argument("--limit-per-split", type=int, default=None)
    args = ap.parse_args()

    s2_root, s1_root = Path(args.s2_root), Path(args.s1_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Indexing S2 patch directories...")
    s2_map = index_patch_dirs(s2_root, "B02")
    print(f"Found {len(s2_map):,} S2 patches")
    print("Indexing S1 patch directories...")
    s1_map = index_patch_dirs(s1_root, "VV")
    print(f"Found {len(s1_map):,} S1 patches")

    df = pd.read_parquet(args.metadata)
    required = {"patch_id", "labels", "split", "s1_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.parquet is missing columns: {sorted(missing)}")

    counts = {}
    for split_name in ["train", "validation", "test"]:
        split = "val" if split_name == "validation" else split_name
        rows = df[df["split"].astype(str).str.lower() == split_name].copy()
        if args.limit_per_split:
            rows = rows.head(args.limit_per_split)

        records = []
        skipped = 0
        for _, row in rows.iterrows():
            s2_id = str(row["patch_id"])
            s1_id = str(row["s1_name"])
            s2_dir = s2_map.get(s2_id)
            s1_dir = s1_map.get(s1_id)
            if s2_dir is None or s1_dir is None:
                skipped += 1
                continue

            optical_paths = []
            sar_paths = []
            ok = True
            for band in S2_BANDS:
                p = s2_dir / f"{s2_id}_{band}.tif"
                if not p.exists(): ok = False
                optical_paths.append(str(p.resolve()))
            for band in S1_BANDS:
                p = s1_dir / f"{s1_id}_{band}.tif"
                if not p.exists(): ok = False
                sar_paths.append(str(p.resolve()))
            if not ok:
                skipped += 1
                continue

            labels = row["labels"]
            if not isinstance(labels, list):
                labels = list(labels) if hasattr(labels, "__iter__") and not isinstance(labels, str) else [str(labels)]
            records.append({
                "optical_paths": optical_paths,
                "sar_paths": sar_paths,
                "labels": [str(x) for x in labels],
                "patch_id": s2_id,
                "s1_name": s1_id,
            })

        with open(out / f"{split}.json", "w") as f:
            json.dump(records, f, indent=2)
        counts[split] = len(records)
        print(f"{split}: {len(records):,} indexed, {skipped:,} skipped")

    with open(out / "index_metadata.json", "w") as f:
        json.dump({"source": "BigEarthNet v2.0", "s2_bands": S2_BANDS,
                   "s1_bands": S1_BANDS, "counts": counts}, f, indent=2)
    print("Done:", out)

if __name__ == "__main__":
    main()
