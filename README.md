# Tiny AI for Raman classification on STM32N6

This repository reproduces the complete Linear Softmax neural-network workflow
for classifying Raman spectra of TEP, DIMP, and DMMP:

1. build the leakage-safe dataset;
2. train, quantize, and test the network locally;
3. train, quantize, and test the same network through STM32 AI Model Zoo
   Services;
4. benchmark the resulting full-INT8 model on `STM32N6570-DK` through ST Edge
   AI Developer Cloud.

The network is `Flatten(512) -> Dense(3, softmax)`, with 1,539 trainable
parameters. It is initialized from random Glorot weights and trained with Adam;
no scikit-learn coefficients or pretrained weights are imported.

This is a closed-set feasibility study on one small, single-source dataset. It
is **not** a validated chemical-agent detector, cannot reject unknown
substances, does not estimate concentration, and must not be used for
operational, medical, or safety decisions.

## Expected results

The untouched test split contains 332 spectra at the held-out 6% and 75%
concentrations.

| Workflow | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Local | Keras FP32 | 99.70% | 99.70% |
| Local | TFLite FP32 | 99.70% | 99.70% |
| Local | TFLite INT8 | 99.70% | 99.70% |
| Model Zoo / N6 candidate | Keras FP32 | 100.00% | 100.00% |
| Model Zoo / N6 candidate | TFLite INT8 | 100.00% | 100.00% |

The Developer Cloud benchmark of the Model Zoo INT8 network on
`STM32N6570-DK` produced 0.02 ms inference time, 0.017 M cycles, 0.53 KiB total
RAM, and 24.35 KiB total flash.

The 100% value is the test-set accuracy of the Model Zoo model prepared for
STM32N6. Developer Cloud benchmarks that exact model on ST's board farm and
reports hardware performance; it does not calculate classification accuracy
from the Raman dataset.

## 1. Clone the repository

Run every shell command below from the repository root unless a section
explicitly changes directory.

```bash
git clone https://github.com/andreaspagnolo/tiny_ai_for_chemical_dangerous_weapon_gas_classification.git
cd tiny_ai_for_chemical_dangerous_weapon_gas_classification
```

## 2. Download the dataset

Download version 1 from the
[Mendeley Data record](https://data.mendeley.com/datasets/jtk7rv77td/1) and
place the three experimental workbooks at exactly these paths:

```text
Raw Raman spectral datasets of TEP, DIMP, and DMMP/
├── TEP_raw_Raman_spectra.xlsx
├── DIMP_raw_Raman_spectra.xlsx
└── DMMP_raw_Raman_spectra.xlsx
```

`Linear_simulated_dataset.xlsx` is deliberately excluded. The source dataset
is not redistributed and remains under the Mendeley CC BY-NC 4.0 licence.

## 3. Build the dataset and reproduce the local 99.70% result

Python 3.12 is required. The Python package versions used for the reference
run are pinned in `requirements.txt`.

```bash
conda create -n raman-local python=3.12.9 -y
conda activate raman-local
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

raman-stm32 inspect
raman-stm32 prepare
raman-linear-softmax all
```

Dataset preparation writes 1,445 training rows, 346 validation rows, and 332
test rows under `data/processed/`. The deterministic split uses seed
`20260805` and groups spectra globally by concentration so that one
concentration cannot occur in multiple splits. Each spectrum is transformed
with per-spectrum standard normal variate, clipped at +/-8 standard deviations,
and scaled to `[-1, 1]`.

`raman-linear-softmax all` trains from scratch, exports FP32 and full-INT8
TFLite models, evaluates all exports on the held-out test set, and writes
`artifacts/reports/linear_softmax_evaluation_metrics.json`.

## 4. Create an ST Edge AI Developer Cloud account

1. Open [ST Edge AI Developer Cloud](https://stedgeai-dc.st.com/).
2. Select **START NOW** or **Sign in**.
3. If you do not have a myST account, select **Create Account**, complete the
   registration form, and finish the requested account verification.
4. **Before running any Developer Cloud command**, sign in through the browser,
   complete the first-access setup, and accept the terms and conditions shown
   by ST.

This browser step is mandatory even when the command-line login reports a
successful connection. If the terms and conditions have not yet been accepted,
the benchmarking command may fail with HTTP 502/503 responses.

Account creation and Developer Cloud access are free of charge. The current
official instructions are available in the
[ST Developer Cloud getting-started guide](https://wiki.st.com/stm32mcu/wiki/AI%3AGetting_started_with_ST_Edge_AI_Developer_Cloud).

The benchmark command later asks for the myST credentials interactively. Do
not store credentials, passwords, or tokens in this repository, its YAML files,
environment variables, or shell commands. Only the TFLite model is uploaded;
the Raman dataset is not uploaded.

## 5. Install the exact STM32 AI Model Zoo Services version

The reference run used STM32 AI Model Zoo Services `v4.1.1`, commit
`0f6210ed5156126b782e1c43249063a477484b20`, and Python 3.12.9. Use a separate
environment from the local workflow.

```bash
export RAMAN_PROJECT_ROOT="$(pwd)"

git clone --branch v4.1.1 --depth 1 \
  https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git \
  ../stm32ai-modelzoo-services
git -C ../stm32ai-modelzoo-services checkout \
  0f6210ed5156126b782e1c43249063a477484b20

conda create -n st-zoo-411 python=3.12.9 -y
conda activate st-zoo-411
python -m pip install -r ../stm32ai-modelzoo-services/requirements.txt

python scripts/install_model_zoo_overlay.py \
  --model-zoo-dir ../stm32ai-modelzoo-services
```

The overlay registers only `raman_linear_softmax` in the official
`arc_fault_detection` service. It also applies the required v4.1.1 fixes for a
static batch-one, full-INT8 TFLite deployment model.

## 6. Reproduce the Model Zoo 100% result

Keep `st-zoo-411` active. The commands below train a new Model Zoo checkpoint,
evaluate it in FP32, quantize it to full INT8, and evaluate the INT8 model on
the same untouched test split.

```bash
cd ../stm32ai-modelzoo-services/arc_fault_detection

python stm32ai_main.py \
  --config-path "$RAMAN_PROJECT_ROOT/configs/stm32_model_zoo" \
  --config-name linear_softmax_training_config.yaml

python stm32ai_main.py \
  --config-path "$RAMAN_PROJECT_ROOT/configs/stm32_model_zoo" \
  --config-name linear_softmax_evaluation_float_config.yaml

python stm32ai_main.py \
  --config-path "$RAMAN_PROJECT_ROOT/configs/stm32_model_zoo" \
  --config-name linear_softmax_quantization_config.yaml

python stm32ai_main.py \
  --config-path "$RAMAN_PROJECT_ROOT/configs/stm32_model_zoo" \
  --config-name linear_softmax_evaluation_int8_config.yaml
```

The generated deployment model is:

```text
artifacts/model_zoo/linear_softmax/quantization/quantized_models/quantized_model.tflite
```

It has signed INT8 input/output, static input shape `[1, 1, 512, 1]`, and only
`RESHAPE`, `FULLY_CONNECTED`, and `SOFTMAX` operators.

## 7. Benchmark the generated network on STM32N6 Developer Cloud

Make sure that the browser login and acceptance of the Developer Cloud terms
and conditions described in section 4 have been completed for the same myST
account.

Keep `st-zoo-411` active and remain in the Model Zoo
`arc_fault_detection` directory.

```bash
python stm32ai_main.py \
  --config-path "$RAMAN_PROJECT_ROOT/configs/stm32_model_zoo" \
  --config-name linear_softmax_benchmarking_stm32n6_config.yaml
```

Enter the myST credentials only at the interactive prompt. The command uploads
the generated INT8 network and benchmarks it on `STM32N6570-DK`. Its output log
is written to:

```text
artifacts/model_zoo/linear_softmax/benchmarking_stm32n6/stm32ai_main.log
```

The reference run was performed on 6 August 2026. Developer Cloud selected
platform 4.0.1 with STM32 backend 12.0.1 after warning that the requested 4.0.0
platform was unavailable.

## 8. Final result check

After all previous commands complete, run this single final check:

```bash
cd "$RAMAN_PROJECT_ROOT"
conda activate raman-local
python scripts/verify_reproduction.py
```

It recalculates the Model Zoo FP32 and INT8 predictions, checks the local
report, validates the deployment tensor/operator contract, and confirms that
the STM32N6 Developer Cloud benchmark completed with the expected result. It
must print:

```text
Reproduction verified successfully
Local Linear Softmax: accuracy 99.70%, macro F1 99.70%
STM32N6 Model Zoo candidate: FP32/INT8 accuracy 100.00%, macro F1 100.00%
Developer Cloud STM32N6570-DK: 0.02 ms, 0.017 M cycles
```

TensorFlow floating-point details can vary across operating systems and CPU
implementations. If the predictions or reported metrics differ, the final
command fails rather than treating the run as an exact reproduction.

## Licence

Except where otherwise stated, the original material in this repository is
licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License. Third-party software, models, datasets, and trademarks
retain their respective licences. See [LICENSE.md](LICENSE.md).

Copyright © 2026 Andrea Spagnolo, Danilo Pau, and STMicroelectronics S.r.l.
