# Tiny AI for Raman classification on STM32

This repository reproduces the complete Linear Softmax neural-network workflow
for classifying Raman spectra of TEP, DIMP, and DMMP:

1. build a leakage-safe dataset;
2. train, export, and test the network locally;
3. train, quantize, and test the same network with STM32 AI Model Zoo Services;
4. benchmark its full-INT8 deployment model on STM32N6, STM32U5, and STM32F4.

The network is `Flatten(512) -> Dense(3, softmax)`, trained from random Glorot
weights with Adam. It has 1,539 parameters and uses 1,445 training spectra:
`1,445 / 1,539 = 0.939` training samples per parameter. This low ratio and the
single-source dataset make this a closed-set feasibility study, not a validated
chemical-agent detector. Do not use it for operational, medical, or safety
decisions.

## Expected results

The untouched test split contains 332 spectra at the held-out 6% and 75%
concentrations.

| Workflow | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Local | Keras FP32 | 99.70% | 99.70% |
| Local | TFLite FP32 | 99.70% | 99.70% |
| Local | TFLite INT8 | 99.70% | 99.70% |
| Model Zoo | Keras FP32 | 100.00% | 100.00% |
| Model Zoo | TFLite INT8 | 100.00% | 100.00% |

The two local FP32 rows are intentional: Keras tests the trained network, while
TFLite FP32 confirms that exporting it does not change its predictions. The
INT8 row then measures the effect of deployment quantization.

The same Model Zoo INT8 file produced these Developer Cloud results:

| Family | Developer Cloud board | Time | Cycles | RAM | Flash |
|---|---|---:|---:|---:|---:|
| STM32N6 | `STM32N6570-DK` | 0.02 ms | 0.017 M | 0.53 KiB | 24.35 KiB |
| STM32U5 | `B-U585I-IOT02A` | 0.06 ms | 0.01 M | 2.03 KiB | 7.94 KiB |
| STM32F4 | `NUCLEO-F401RE` | 0.11 ms | 0.009 M | 2.03 KiB | 7.92 KiB |

N6 is fastest, but this 1,539-parameter network is small enough that U5 and F4
are practical alternatives with lower reported flash use. Developer Cloud
measures hardware performance; the 100% classification accuracy is calculated
separately on the Raman test split.

## 1. Clone and select the project root

Commands are run from the repository root. All commands below are one-line
commands valid in macOS/Linux terminals and Windows Anaconda Prompt unless a
platform is explicitly named.

```bash
git clone https://github.com/andreaspagnolo/tiny_ai_for_chemical_dangerous_weapon_gas_classification.git
cd tiny_ai_for_chemical_dangerous_weapon_gas_classification
```

On macOS/Linux:

```bash
export RAMAN_PROJECT_ROOT="$(pwd)"
```

On Windows Anaconda Prompt:

```bat
set "RAMAN_PROJECT_ROOT=%cd%"
```

Set this variable again after opening a new terminal.

## 2. Download the dataset

Download version 1 from the
[Mendeley Data record](https://data.mendeley.com/datasets/jtk7rv77td/1) and put
the three experimental workbooks at these exact paths:

```text
Raw Raman spectral datasets of TEP, DIMP, and DMMP/
├── TEP_raw_Raman_spectra.xlsx
├── DIMP_raw_Raman_spectra.xlsx
└── DMMP_raw_Raman_spectra.xlsx
```

Do not add `Linear_simulated_dataset.xlsx`. The source data is not
redistributed and remains under its Mendeley CC BY-NC 4.0 licence.

## 3. Reproduce the local 99.70% result

Python 3.12.9 and every Python dependency are pinned.

```bash
conda create -n raman-local python=3.12.9 -y
conda activate raman-local
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
raman-stm32 inspect
raman-stm32 prepare
raman-linear-softmax all
```

`prepare` creates 1,445 training, 346 validation, and 332 test rows using seed
`20260805`. Concentrations are globally grouped so none can occur in more than
one split. `all` trains from scratch, exports FP32 and full-INT8 TFLite files,
and evaluates all three local representations.

## 4. Activate ST Edge AI Developer Cloud

1. Open [ST Edge AI Developer Cloud](https://stedgeai-dc.st.com/), select
   **START NOW**, and sign in with a myST account or create one.
2. Complete the account verification and first-access setup.
3. Sign in through the browser and accept the ST terms and conditions before
   running a benchmark.

ST's official [Developer Cloud getting-started guide](https://wiki.st.com/stm32mcu/wiki/AI%3AGetting_started_with_ST_Edge_AI_Developer_Cloud)
illustrates the account creation and login flow.

The third step is mandatory even if command-line authentication succeeds;
without it the service can return HTTP 502/503 errors. Benchmark commands ask
for myST credentials interactively. Do not put credentials or tokens in this
repository, YAML files, environment variables, or shell commands. Only the
TFLite model is uploaded; the Raman dataset is not.

## 5. Install the pinned Model Zoo environment

Leave the local environment first, then create the separate ST environment:

```bash
conda deactivate
conda create -n st-zoo-411 python=3.12.9 -y
conda activate st-zoo-411
python scripts/install_model_zoo_overlay.py --model-zoo-dir ../stm32ai-modelzoo-services --clone
python -m pip install -r ../stm32ai-modelzoo-services/requirements.txt
```

The installer fetches STM32 AI Model Zoo Services `v4.1.1` at exact commit
`0f6210ed5156126b782e1c43249063a477484b20` and applies the project overlay.
The official service itself is not copied into this repository because it
provides the Developer Cloud client and board support; all project-specific
model code, configurations, compatibility fixes, and commands are included
here. The overlay also fixes the v4.1.1 Windows path/MLflow issues and prevents
an unrelated global ClearML configuration from uploading project artifacts.

## 6. Reproduce the Model Zoo 100% result

Keep `st-zoo-411` active and enter the official service directory:

```bash
cd ../stm32ai-modelzoo-services/arc_fault_detection
```

The following four commands respectively train a new Keras checkpoint, test
the FP32 model, convert it to full INT8, and test the INT8 model:

```bash
python stm32ai_main.py --config-path ../../tiny_ai_for_chemical_dangerous_weapon_gas_classification/configs/stm32_model_zoo --config-name linear_softmax_training_config.yaml
python stm32ai_main.py --config-path ../../tiny_ai_for_chemical_dangerous_weapon_gas_classification/configs/stm32_model_zoo --config-name linear_softmax_evaluation_float_config.yaml
python stm32ai_main.py --config-path ../../tiny_ai_for_chemical_dangerous_weapon_gas_classification/configs/stm32_model_zoo --config-name linear_softmax_quantization_config.yaml
python stm32ai_main.py --config-path ../../tiny_ai_for_chemical_dangerous_weapon_gas_classification/configs/stm32_model_zoo --config-name linear_softmax_evaluation_int8_config.yaml
```

The deployment file is written to:

```text
artifacts/model_zoo/linear_softmax/quantization/quantized_models/quantized_model.tflite
```

It has signed INT8 input/output, static input shape `[1, 1, 512, 1]`, and only
`RESHAPE`, `FULLY_CONNECTED`, and `SOFTMAX` operators.

## 7. Reproduce the three Developer Cloud benchmarks

After completing the browser activation in section 4, run:

```bash
python stm32ai_main.py --config-path ../../tiny_ai_for_chemical_dangerous_weapon_gas_classification/configs/stm32_model_zoo --config-name linear_softmax_benchmarking_stm32n6_config.yaml
python stm32ai_main.py --config-path ../../tiny_ai_for_chemical_dangerous_weapon_gas_classification/configs/stm32_model_zoo --config-name linear_softmax_benchmarking_stm32u5_config.yaml
python stm32ai_main.py --config-path ../../tiny_ai_for_chemical_dangerous_weapon_gas_classification/configs/stm32_model_zoo --config-name linear_softmax_benchmarking_stm32f4_config.yaml
```

Enter the myST credentials only when prompted. The command benchmarks the same
INT8 model on N6, U5, and F4 and stores each log under
`artifacts/model_zoo/linear_softmax/benchmarking_<family>/stm32ai_main.log`.
The reference runs used Developer Cloud platform 4.0.1 and STM32 backend
12.0.1 on 6 and 10 August 2026.

## 8. Final result check

After every previous command completes, run this single final check:

```bash
cd ../../tiny_ai_for_chemical_dangerous_weapon_gas_classification
conda deactivate
conda activate raman-local
python scripts/verify_reproduction.py
```

It recalculates both Model Zoo accuracies, checks all local results, validates
the INT8 tensor/operator contract, the sample/parameter counts, and all three
Developer Cloud logs. Exact reproduction prints:

```text
Reproduction verified successfully
Local Linear Softmax: accuracy 99.70%, macro F1 99.70%
Model Zoo Linear Softmax: FP32/INT8 accuracy 100.00%, macro F1 100.00%
Training samples / parameters: 1445 / 1539 = 0.939
Developer Cloud: N6 0.02 ms; U5 0.06 ms; F4 0.11 ms
```

TensorFlow floating-point details can vary across operating systems and CPU
implementations. If predictions or reported values differ, this check fails
instead of accepting an approximate reproduction.

## Licence

Except where otherwise stated, original material in this repository is
licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License. Third-party software, datasets, and trademarks retain
their respective licences. See [LICENSE.md](LICENSE.md).

Copyright © 2026 Andrea Spagnolo, Danilo Pau, and STMicroelectronics S.r.l.
