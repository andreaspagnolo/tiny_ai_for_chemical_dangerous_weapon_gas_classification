# Tiny AI for chemical dangerous weapon gas classification

This repository contains a reproducible, safety-oriented feasibility study for
classifying Raman spectra of three lower-hazard organophosphorus simulants:
TEP, DIMP, and DMMP. It prepares the published dataset, trains a 1,539-parameter
Linear Softmax neural network from random weights, exports FP32 and full-integer
INT8 TensorFlow Lite models, and evaluates them on a held-out test split.

This is a closed-set classifier for the three supplied classes. It is **not** a
validated chemical-agent detector, does not reject unknown substances, does
not estimate concentration, and must not be used for operational, medical, or
safety decisions.

## Expected result

With the dataset and pinned Python dependencies described below, the held-out
332-spectrum test split should produce the following results:

| Export | Accuracy | Macro F1 |
|---|---:|---:|
| Keras FP32 | 99.70% | 99.70% |
| TFLite FP32 | 99.70% | 99.70% |
| TFLite INT8 | 99.70% | 99.70% |

The FP32 and INT8 TFLite models should agree on all test predictions. The INT8
model should have static batch size 1, INT8 input/output, 1,539 parameters, and
only `RESHAPE`, `FULLY_CONNECTED`, and `SOFTMAX` operators.

These are internal holdout results from one small, single-source dataset. They
do not establish generalization to other instruments, backgrounds, sample
matrices, compounds, or field conditions.

## 1. Clone the repository

The shell commands in this README must be run from the repository root, **not**
from inside the raw-dataset directory.

```bash
git clone https://github.com/andreaspagnolo/tiny_ai_for_chemical_dangerous_weapon_gas_classification.git
cd tiny_ai_for_chemical_dangerous_weapon_gas_classification
```

## 2. Download and place the dataset

Download version 1 of the dataset from the
[Mendeley Data record](https://data.mendeley.com/datasets/jtk7rv77td/1). Create
the directory below at the repository root and place the three experimental
workbooks in it:

```text
Raw Raman spectral datasets of TEP, DIMP, and DMMP/
├── TEP_raw_Raman_spectra.xlsx
├── DIMP_raw_Raman_spectra.xlsx
└── DMMP_raw_Raman_spectra.xlsx
```

`Linear_simulated_dataset.xlsx` and the publisher's `README.txt` may remain in
the same directory, but the simulated workbook is deliberately excluded from
the classification task. The original dataset is not redistributed by this
repository and remains under the Mendeley CC BY-NC 4.0 licence.

Before continuing, check that the required files are in the correct location:

```bash
test -f "Raw Raman spectral datasets of TEP, DIMP, and DMMP/TEP_raw_Raman_spectra.xlsx"
test -f "Raw Raman spectral datasets of TEP, DIMP, and DMMP/DIMP_raw_Raman_spectra.xlsx"
test -f "Raw Raman spectral datasets of TEP, DIMP, and DMMP/DMMP_raw_Raman_spectra.xlsx"
```

All three commands should finish silently with exit status zero.

## 3. Create the pinned Conda environment

Python 3.12 is required. The dependency versions used to generate the supplied
results are pinned in `requirements.txt`.

```bash
conda create -n raman-preprocessing python=3.12 -y
conda activate raman-preprocessing
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Confirm that the package and all Linear Softmax helper modules are available:

```bash
python -c "import raman_stm32.evaluation, raman_stm32.modeling, raman_stm32.samples, raman_stm32.verification; print('installation OK')"
python -m pytest -q
```

The test suite should report `4 passed`. If Python reports
`No module named 'raman_stm32.evaluation'`, the checkout is incomplete: the
four helper files under `src/raman_stm32/` are required, and reinstalling an
incomplete checkout cannot create them.

## 4. Reproduce the dataset and model results

Keep the `raman-preprocessing` environment active and remain at the repository
root. Run the complete sequence exactly once:

```bash
raman-stm32 inspect
raman-stm32 prepare
raman-linear-softmax all
```

The equivalent explicit sequence is useful when inspecting one phase at a
time. Do not run both sequences unless you intentionally want to retrain and
overwrite the generated artifacts.

```bash
raman-stm32 inspect
raman-stm32 prepare
raman-linear-softmax train
raman-linear-softmax quantize
raman-linear-softmax evaluate
raman-linear-softmax export-samples
raman-linear-softmax verify
```

Training starts from deterministic random Glorot weights with seed `20260805`.
No scikit-learn or pretrained coefficients are imported. Quantization uses 384
training spectra for calibration, and evaluation uses only the untouched test
split.

TensorFlow numerical details can vary slightly between operating systems and
CPU implementations. The classification metrics and prediction agreement
below are the reproducibility criteria; byte-for-byte identity of regenerated
Keras files is not promised.

## 5. Verify the reproduced metrics

After the pipeline finishes, copy and run this check from the repository root:

```bash
python - <<'PY'
import json
from pathlib import Path

report_path = Path("artifacts/reports/linear_softmax_evaluation_metrics.json")
report = json.loads(report_path.read_text(encoding="utf-8"))
expected_accuracy = 0.9969879518072289
expected_macro_f1 = 0.9970209513356721

for model_name in (
    "linear_softmax_float32_keras",
    "linear_softmax_float32_tflite",
    "linear_softmax_int8",
):
    metrics = report["metrics"][model_name]
    assert abs(metrics["accuracy"] - expected_accuracy) < 1e-12, metrics
    assert abs(metrics["macro_f1"] - expected_macro_f1) < 1e-12, metrics

agreement = report["linear_float_int8"]["prediction_agreement"]
assert agreement == 1.0, agreement
print("Reproduction verified: 99.70% accuracy, 99.70% macro F1, FP32/INT8 agreement 100%")
PY
```

## Generated outputs

Dataset preparation writes:

```text
data/processed/
├── train.csv             # 1,445 spectra
├── validation.csv        # 346 spectra
├── test.csv              # 332 spectra
├── prediction.csv
├── metadata.csv
├── raman_shift_axis.csv
└── dataset.npz
```

Training and export write the models to `artifacts/models/`, detailed JSON
reports to `artifacts/reports/`, and six deployment examples to
`artifacts/samples/linear_softmax/`.

For predictions on the prepared six-row example:

```bash
raman-linear-softmax predict \
  --input-csv data/processed/prediction.csv \
  --preprocessed
```

Omit `--preprocessed` only for a headerless CSV containing raw 512-point Raman
spectra. Further details are in [the dataset documentation](docs/DATASET.md),
[the experiment report](docs/LINEAR_SOFTMAX_EXPERIMENT.md), and
[the model card](docs/LINEAR_SOFTMAX_MODEL_CARD.md).

## Dataset preparation policy

The preparation step verifies the shared 512-point Raman axis and finite
values, retains compound/sample/concentration metadata, and applies
per-spectrum standard normal variate (SNV), clipping at ±8 standard deviations,
then scaling to `[-1, 1]`.

The published files contain no session or specimen identifiers. Concentration
is therefore used as a conservative global grouping proxy so that spectra at
the same concentration never appear in different splits. The split is
deterministic and does not use spectral feature values.

## Compounds and research context

- **TEP — triethyl phosphate:** used in applications including flame
  retardancy and as a lower-hazard simulant in controlled protective-material
  research.
- **DIMP — diisopropyl methylphosphonate:** an organophosphonate used in
  controlled analytical, degradation, and instrument-calibration studies.
- **DMMP — dimethyl methylphosphonate:** an organophosphonate widely used as a
  lower-hazard simulant in sensor and environmental-detection research.

The compounds still require appropriate laboratory safety procedures.

## Ownership and licence

Except where otherwise stated, the original material in this repository is
licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License (CC BY-NC-SA 4.0).

Copyright © 2026 Andrea Spagnolo, Danilo Pau, and STMicroelectronics S.r.l.

See [LICENSE.md](LICENSE.md) for the complete terms. Third-party software,
models, datasets, images, trademarks, and external assets retain their own
licences and are not covered by the repository licence unless explicitly
stated.
