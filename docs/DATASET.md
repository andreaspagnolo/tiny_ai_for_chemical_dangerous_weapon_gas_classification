# Dataset audit and split rationale

## Source and scope

Dataset: **Raw Raman spectral datasets of TEP, DIMP, and DMMP with a linear
simulated dataset**, version 1, DOI
[10.17632/jtk7rv77td.1](https://doi.org/10.17632/jtk7rv77td.1), contributor
Fanghui Zhong. The Mendeley record identifies the licence as CC BY-NC 4.0.

Only these experimental files enter the classification task:

- `TEP_raw_Raman_spectra.xlsx`: 683 rows.
- `DIMP_raw_Raman_spectra.xlsx`: 723 rows.
- `DMMP_raw_Raman_spectra.xlsx`: 717 rows.

`Linear_simulated_dataset.xlsx` is not loaded. The common experimental axis has
512 strictly increasing values from 36.8439758886 to 2546.0840049782 cm^-1.
All loaded values are finite.

## Concentration units

The experimental cells contain values from `0.005` through `1`. The publisher's
README describes these as the 0.5% through 100% levels. The pipeline therefore
preserves two explicit metadata fields:

- `concentration_fraction`: the cell value, unchanged;
- `concentration_percent`: cell value multiplied by 100.

Concentration is used for splitting and reporting, never as an input feature.

## Group split

There is no acquisition/session/specimen identifier in the published files;
`Number` is described as a spectrum index. A random row split could therefore
place likely replicates from the same prepared concentration into multiple
sets. To avoid that leakage, `concentration_fraction` is used as a conservative
global group: all TEP, DIMP, and DMMP spectra at one concentration are assigned
together.

The deterministic enumeration selects 7/2/2 groups with validation and test
each containing one low and one high concentration. It minimizes sample-count
and class-count deviation from 70/15/15 using metadata only. The final split is
listed in the root README and fully serialized in `dataset_report.json`.

There are 41 occurrences belonging to 20 exact-duplicate spectral patterns.
Every pattern is contained in a single concentration and the audit confirms
zero patterns crossing splits. Missing row indices (TEP 319 and DMMP 710) are
reported but are not imputed.

## Preprocessing

For each 512-point spectrum `x`:

```text
z = (x - mean(x)) / std_population(x)
input = clip(z, -8, 8) / 8
```

The output is float32 in `[-1, 1]`. This per-spectrum transform needs no fitted
statistics, so the exact same operation is available at inference. The C
reference is in `firmware/raman_preprocess.c`.

## Reproducibility outputs

- `data/processed/dataset.npz`: arrays and non-sensitive metadata for the local
  Python pipeline.
- `data/processed/{train,validation,test}.csv`: headerless, preprocessed
  512-feature rows plus the final integer label, matching the Model Zoo signal
  loader.
- `data/processed/metadata.csv`: sample ID, class, fraction/percent
  concentration, group, source, and split.
- `data/processed/raman_shift_axis.csv`: exact input-axis order.

Generated data are ignored by version control. Recreate them from the licensed
source files with `raman-stm32 prepare`.

