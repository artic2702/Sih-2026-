# Audited Person-4 Fusion Package

Included fixes beyond the uploaded baseline/Claude deliverable:

- official BigEarthNet v2 separate-band index preparation
- BigEarthNet v2 optical/SAR loader support
- corrected selected-band handling for B02/B03/B04/B08 + VV/VH
- corrected Sentinel-1 v2 dB preprocessing (no second log transform)
- corrected CUDA device placement for all fusion modules
- corrected ablation batch-size iteration
- best validation checkpoint copied to primary `model.pt`
- cleaned fusion notebook; removed hardcoded fusion>=optical assertion
- lightweight Colab dependency file
- exact Colab setup instructions

Frozen shared files (`src/common/schemas.py`, `src/common/constants.py`) were not modified.
