# SatQuery AI — Person 4 Fusion: Google Colab Training

This package is the audited Person-4 fusion baseline. It trains the existing **Dual ResNet-18 (4-band Sentinel-2 + 2-band Sentinel-1) + MLP fusion head** and saves a small checkpoint for later web/React integration.

## What is included

- audited repo + Claude fusion deliverable changes
- official BigEarthNet v2 preparation helper (`scripts/prepare_bigearthnet_mm.py`)
- fixed BigEarthNet v2 separate-band loading
- fixed Sentinel-1 dB preprocessing (do not log an already-dB product)
- fixed CUDA model placement
- fixed 3-way ablation iteration
- cleaned `notebooks/04_optical_sar_fusion.ipynb`
- Colab training defaults: batch 32, 5 epochs; notebook uses 3 epochs

## Important

The official BigEarthNet v2 release is distributed as separate Sentinel-1/Sentinel-2 patch archives plus `metadata.parquet`. The helper only builds JSON indexes; it does not duplicate the imagery.

Do not put the full dataset inside this repo zip. Keep the dataset in Google Drive/Colab storage.

## Expected final artifact

After training:

`models/checkpoints/fusion/fusion_head/model.pt`

This is the file that should later be shipped with the application/backend and loaded for inference. The React site should call a backend inference endpoint; it should not run PyTorch training in the browser.
