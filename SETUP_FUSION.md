# Optical-SAR Fusion — setup & run order

## 1. Drop these files into your repo (same paths, overwrite)
- `scripts/generate_synthetic_bigearthnet_mm.py`  (new)
- `scripts/run_fusion_ablation.py`  (new)
- `src/training/train_fusion.py`  (overwrite — was a stub)
- `src/agent/tool_registry.py`  (overwrite — fixes model.load() never being called)
- `src/agent/controller.py`  (overwrite — fixes predict() never being called + real preprocessing)

`src/models/fusion_model.py` was already complete/correct — untouched.

## 2. Generate training data (skip if real BigEarthNet-MM finishes downloading in time)
```bash
python scripts/generate_synthetic_bigearthnet_mm.py --out data/raw/bigearthnet_mm \
    --n-train 240 --n-val 48 --n-test 48
```
Swap in the real dataset later with the exact same file layout — nothing else changes.

## 3. Train
```bash
python -m src.training.train_fusion --config configs/config.yaml
# quick smoke run first if you want to confirm nothing crashes before a full run:
python -m src.training.train_fusion --config configs/config.yaml --epochs 2
```
Watch for `val_macro_f1` printed each epoch. Checkpoint lands at
`models/checkpoints/fusion/fusion_head` (+ `.meta.json` + `.best` variant).

## 4. Run the mandatory 3-way ablation
```bash
python scripts/run_fusion_ablation.py --config configs/config.yaml --split test
```
Prints the table to console and writes:
- `src/evaluation/results/fusion_ablation.json`
- `src/evaluation/results/fusion_ablation.md`  ← paste this table straight into your pitch deck.

## 5. Confirm it's live in the app
```bash
python -m scripts.smoke_test          # fusion line should now show [PASS]
streamlit run app/app.py              # upload optical + SAR images, ask a fusion query
```

## Known issues found while wiring this up — not fixed here, flag to the team
1. **`configs/config.yaml`'s `isro_sac.sar_bands` has only 1 band listed**, but the
   fusion encoder is built expecting 2 SAR channels (from `bigearthnet_mm`'s config).
   Controller.py works around this by always preprocessing with `bigearthnet_mm`'s
   band config, but `isro_sac`'s entry itself is still wrong and needs a real fix
   before judging — it's marked "TODO: confirm actual RISAT band layout" in the
   config, so this was already a known unknown, just confirming it's a real blocker now.
2. **`src/evaluation/evaluate.py`'s fusion path has a gold-label bug**: it collects
   raw class-name-string lists into `gold_labels` instead of converting to the
   multi-hot binary array `multilabel_f1()` needs. `run_fusion_ablation.py` avoids
   this file entirely and does its own (correct) collection — but `evaluate.py`
   itself is still broken for anyone who calls `python -m src.evaluation.evaluate --task fusion`.
3. **Single-image and bi-temporal preprocessing are still not wired into
   `controller.py`** — I only implemented the cross_modal (fusion) path, since
   that's the one I own. VQA/captioning/grounding and change will still fail with
   a clear `NotImplementedError` until Person 1/2/3 do the same wiring for their
   models' expected input format.
