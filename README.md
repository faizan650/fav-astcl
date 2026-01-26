# FAV-ASTCL

Forecasting-Aware and Versatile Adaptive Spatio-Temporal Context Learning (FAV-ASTCL)

This repository contains the implementation of FAV-ASTCL, an extension of ASTCL for
adaptive, forecasting-aware traffic prediction on spatio-temporal graphs.

## Features
- Learnable adaptive context selection
- Sparse dynamic graph modeling
- Horizon-aware loss optimization
- Evaluated on METR-LA and PEMS-BAY

## Datasets
Public traffic datasets (METR-LA, PEMS-BAY) are not included due to size constraints.
Please download them separately and place them under `datasets/`.

## Usage
```bash
python src/train.py
