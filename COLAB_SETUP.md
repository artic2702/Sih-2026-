# Colab: exact Person-4 training steps

## 0. Runtime

In Google Colab: **Runtime → Change runtime type → T4 GPU** (or any available NVIDIA GPU).

## 1. Upload and unzip this package

Upload `satquery_fusion_colab.zip` to Colab, then:

```bash
!unzip -q satquery_fusion_colab.zip
%cd satquery_fusion_colab
```

If Colab created a nested directory, `!ls` and `%cd` into the directory containing `configs/` and `src/`.

## 2. Install the lightweight training dependencies

```bash
!pip install -q -r requirements_colab.txt
```

The standard Colab runtime already includes CUDA PyTorch/torchvision. If your runtime does not, install a matching CUDA PyTorch build before continuing.

## 3. Confirm the GPU

```bash
!nvidia-smi
```

Then:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
```

You want `True` for CUDA.

## 4. Put the BigEarthNet v2 data somewhere accessible

Recommended: Google Drive or Colab local storage. The official BigEarthNet v2 release contains separate `BigEarthNet-S2`, `BigEarthNet-S1`, and `metadata.parquet` resources.

The expected extracted layout is approximately:

```text
BigEarthNet-S2/
  <S2_TILE>/
    <S2_PATCH>/
      <S2_PATCH>_B02.tif
      <S2_PATCH>_B03.tif
      <S2_PATCH>_B04.tif
      <S2_PATCH>_B08.tif
      ...

BigEarthNet-S1/
  <S1_TILE>/
    <S1_PATCH>/
      <S1_PATCH>_VV.tif
      <S1_PATCH>_VH.tif
      ...

metadata.parquet
```

If your data is in Google Drive, mount it first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 5. Build the lightweight index

Do NOT copy the imagery into the repo. The helper only writes `train.json`, `val.json`, and `test.json` containing paths + labels.

For a first sanity check, use a small subset:

```bash
!python scripts/prepare_bigearthnet_mm.py \
  --s2-root /content/BigEarthNet-S2 \
  --s1-root /content/BigEarthNet-S1 \
  --metadata /content/metadata.parquet \
  --out data/raw/bigearthnet_mm \
  --limit-per-split 2000
```

If the real dataset is on Drive, replace the paths with `/content/drive/MyDrive/...`.

**Before a longer run, inspect the printed counts.** You want non-zero train/val/test counts. If almost everything is skipped, stop and inspect your dataset paths instead of training.

## 6. Smoke-test one batch

```bash
!python - <<'PY'
import yaml
from src.preprocessing.dataset_loader import get_dataloader

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

loader = get_dataloader(
    task='fusion', split='train', dataset='bigearthnet_mm',
    config=config, batch_size=4
)
b = next(iter(loader))
print('optical:', b['image_optical'].shape)
print('sar:', b['image_sar'].shape)
print('labels:', b['label'][:2])
PY
```

Expected shapes:

```text
optical: torch.Size([4, 4, 256, 256])
sar:     torch.Size([4, 2, 256, 256])
```

## 7. Start training

For the first real run, use 3 epochs:

```bash
!python -m src.training.train_fusion --config configs/config.yaml --epochs 3
```

The model attempts to download ImageNet-pretrained ResNet-18 weights. Colab normally has internet access, so this should succeed. **Do not use the random-initialization fallback as a reportable benchmark.**

The training script freezes both ResNet encoders and trains only the MLP fusion head, as required by the project plan.

## 8. Check the checkpoint

After training:

```bash
!ls -lh models/checkpoints/fusion/fusion_head/
```

The important file is:

```text
models/checkpoints/fusion/fusion_head/model.pt
```

The script keeps the best validation checkpoint as the primary `model.pt`.

## 9. Run the mandatory ablation

```bash
!python scripts/run_fusion_ablation.py \
  --config configs/config.yaml \
  --split test
```

Outputs:

```text
src/evaluation/results/fusion_ablation.json
src/evaluation/results/fusion_ablation.md
```

The three legs are:

1. optical-only
2. SAR-only
3. optical + SAR fusion

Do not edit the scores or force fusion to win.

## 10. Download the trained checkpoint

```python
from google.colab import files
files.download('models/checkpoints/fusion/fusion_head/model.pt')
```

Also download the ablation JSON/Markdown if your team needs the results:

```python
files.download('src/evaluation/results/fusion_ablation.json')
files.download('src/evaluation/results/fusion_ablation.md')
```

## 11. What goes into the React application later

The React frontend should **not** contain or run the PyTorch model. The intended architecture is:

```text
React site
   ↓ HTTP request
Python backend / inference service
   ↓
FusionModel + model.pt
   ↓
RSModelResult JSON
   ↓
React UI
```

The trained `model.pt` is the artifact we need to carry forward.

## 12. If the dataset path/format does not match

Do not start changing random model code. First run:

```bash
!python scripts/prepare_bigearthnet_mm.py --help
```

and verify that the official v2 S1/S2 directories and `metadata.parquet` are the paths you supplied. The preparation script is specifically for the official BigEarthNet v2 separate-band layout.
