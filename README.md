# DBUMIAD-ER
Erasure-aware diffusion model for unsupervised medical image anomaly detection on chest X-rays.

## Setup
1. Install PyTorch for your CUDA version from pytorch.org
2. pip install -r requirements.txt
3. Place datasets as described in data/README.md

## Structure
configs/   one YAML per experiment
src/       data, models, utils
scripts/   preprocess, train, infer, evaluate
results/   metrics (CSV) and figures
