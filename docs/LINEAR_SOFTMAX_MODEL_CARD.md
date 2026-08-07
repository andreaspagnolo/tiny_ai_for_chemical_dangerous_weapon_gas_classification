# Model card: raman_linear_softmax

## Intended use

STM32N6 feasibility study for closed-set classification of the supplied TEP,
DIMP and DMMP Raman spectra. This is not a
validated chemical-agent detector and must not be used for operational,
medical, environmental or safety decisions.

## Architecture and training origin

The model is `Flatten(512) -> Dense(3, softmax)` with 1,539 trainable
parameters and L2 regularization of `1e-4`. It was trained from random Glorot
weights using Adam. No weight from the logistic-regression baseline was
imported.

The deployment input is static `[1,1,512,1]`; outputs are ordered
`[TEP,DIMP,DMMP]`. The standalone INT8 tensor uses input scale
`0.004478252027183771`, zero point `-96`, and output scale `0.00390625`, zero
point `-128`.

## Evaluation

The standalone FP32 and INT8 models achieved 99.70% accuracy and 99.70% macro
F1 on 332 spectra from the held-out 6% and 75% concentration groups.
Quantization preserved all predictions.

## Limitations

- one published data source and instrument setup;
- no independent laboratory, instrument, day, operator or matrix;
- no blanks, interferents, mixtures, unknown or out-of-distribution class;
- concentration is only a grouping proxy because acquisition IDs are absent;
- only two concentration groups occur in the test set;
- no physical-board evaluation or per-layer Neural-ART mapping report has yet
  been recorded.

Consequently, the strong internal accuracy primarily demonstrates feasibility
and linear separability of this prepared dataset.
