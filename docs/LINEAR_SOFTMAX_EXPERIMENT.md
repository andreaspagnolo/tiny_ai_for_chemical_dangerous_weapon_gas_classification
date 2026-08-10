# Train-from-scratch Linear Softmax experiment

## Purpose

This experiment tests whether the performance of multinomial logistic
regression can be retained in a small Keras/TFLite neural network.

The model is a genuine neural-network training run:

```text
Input [1, 1, 512, 1] -> Flatten [512] -> Dense [3] -> Softmax
```

Although this has the same mathematical model family as multinomial logistic
regression, its 1,539 parameters are initialized with random Glorot weights
and learned with Adam. No coefficient, intercept, layer or checkpoint is
copied from the scikit-learn baseline.

The L2 value `1e-4` was chosen using train and validation data only. The test
set remained excluded from this choice and was evaluated after the training
configuration had been fixed.

## Training and standalone results

- seed: `20260805`;
- optimizer: Adam, learning rate `0.01`;
- loss: sparse categorical cross-entropy plus L2 regularization;
- maximum epochs: 300;
- early stopping: minimum validation loss, patience 40;
- training samples: 1,445;
- training-sample/parameter ratio: `1,445 / 1,539 = 0.939`;
- validation samples: 346;
- test samples: 332 from the held-out 6% and 75% concentration groups;
- epochs completed: 285;
- selected epoch: 246;
- selected validation accuracy: 100%.

| Model | Test accuracy | Macro F1 | Size |
|---|---:|---:|---:|
| Linear Softmax Keras FP32 | 99.70% | 99.70% | 22,207 B |
| Linear Softmax TFLite FP32 | 99.70% | 99.70% | 7,752 B |
| Linear Softmax TFLite INT8 | 99.70% | 99.70% | 3,424 B |

The Linear Softmax confusion matrix is `[[98,1,0],[0,129,0],[0,0,104]]`.
Accuracy is 99.40% at 6% concentration and 100% at 75%. INT8 changes no class
prediction relative to TFLite FP32; agreement is 100%, and the mean absolute
score difference is 0.00278.

These results indicate that the prepared spectra are close to linearly
separable. The sub-one training-sample/parameter ratio also reinforces that
the result is a limited feasibility result, not evidence of broad
generalization. It does not establish performance on other instruments,
laboratories, matrices, interferents, unknown substances or field samples.

## Embedded properties

- 1,539 parameters;
- approximately 1,536 multiply-accumulates per spectrum;
- static batch-one input `[1,1,512,1]`;
- static three-score output `[1,3]`;
- signed INT8 input and output;
- TFLite operators: `RESHAPE`, `FULLY_CONNECTED`, `SOFTMAX`;
- standalone INT8 input scale `0.0044782520`, zero point `-96`;
- output scale `0.00390625`, zero point `-128`.

The operator inventory is intentionally simple. Final Neural-ART mapping still
requires a separate STEdgeAI compiler report and is outside this PC-only
training/evaluation commit.

## Commands

Standalone experiment:

```bash
PYTHONPATH=src python -m raman_stm32.linear_softmax_experiment all
PYTHONPATH=src python -m raman_stm32.linear_softmax_experiment predict \
  --input-csv data/processed/prediction.csv --preprocessed
```
