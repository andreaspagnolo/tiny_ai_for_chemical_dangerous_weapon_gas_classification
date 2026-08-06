# Raman dataset preparation — phase 1

This first repository phase contains only the reusable code and configuration
for inspecting and preparing the experimental Raman spectra of TEP, DIMP and
DMMP. Training, quantization, STM32 deployment files, generated artifacts and
the original workbooks are intentionally excluded from this commit.

The source dataset is version 1 of the Mendeley record:
<https://data.mendeley.com/datasets/jtk7rv77td/1>

Download the three experimental workbooks and place them in:

```text
Raw Raman spectral datasets of TEP, DIMP, and DMMP/
├── TEP_raw_Raman_spectra.xlsx
├── DIMP_raw_Raman_spectra.xlsx
└── DMMP_raw_Raman_spectra.xlsx
```

`Linear_simulated_dataset.xlsx` is deliberately excluded from the main task.
The original data are not redistributed and remain subject to the Mendeley
CC BY-NC 4.0 licence.

## Reproduce the preparation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

raman-stm32 inspect
raman-stm32 prepare
python -m pytest -q
```

The preparation code:

- verifies the common 512-point Raman axis and finite values;
- retains compound, sample number and concentration metadata;
- excludes the simulated linear workbook;
- assigns complete concentration groups to train, validation or test;
- uses a deterministic seed (`20260805`);
- applies per-spectrum SNV, clipping to ±8 standard deviations and scaling to
  `[-1, 1]`;
- writes Model-Zoo-compatible CSV files with 512 features and one label column.

The published files do not contain session or specimen identifiers. Therefore
concentration is used as a conservative global group proxy to prevent spectra
from the same concentration appearing in different splits.

Generated files are written under `data/processed/` and remain ignored by Git:

```text
data/processed/
├── train.csv
├── validation.csv
├── test.csv
├── prediction.csv
├── metadata.csv
├── raman_shift_axis.csv
└── dataset.npz
```

The split and preprocessing rationale is documented in
[`docs/DATASET.md`](docs/DATASET.md). Later commits can add the model and STM32
Model Zoo phases without changing this preparation contract.
