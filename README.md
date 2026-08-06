# Tiny AI for chemical dangerous weapon gas classification

## Project context for non-specialists

This repository is a research and safety-oriented feasibility study using
Raman spectra of three organophosphorus compounds: TEP, DIMP and DMMP. These
compounds are commonly used as lower-hazard chemical simulants for laboratory,
defence and sensor research related to G-series nerve agents such as Sarin and
Soman. Real nerve agents are extremely toxic and subject to strict
international controls, so controlled simulants allow researchers to test
instruments, detection systems and protective materials without handling the
actual agents.

The compounds still require appropriate laboratory safety procedures. This
project is only a closed-set Raman-spectrum classifier for the three supplied
classes. It is not a validated chemical-agent detector, does not estimate
concentration, does not identify unknown substances and must not be used for
operational, medical or safety decisions.

### Meaning of the abbreviations

- **TEP — triethyl phosphate:** a liquid used, among other applications, as a
  flame retardant. In defence-related experiments it can be used to study how
  a simulant interacts with protective materials through adsorption and
  desorption.
- **DIMP — diisopropyl methylphosphonate:** an organophosphonate associated
  with Sarin-related chemistry and used in controlled studies of degradation,
  analytical instrumentation and alarm calibration.
- **DMMP — dimethyl methylphosphonate:** a colourless organophosphonate used in
  flame-retardant applications and widely used as a Sarin simulant in sensor
  and environmental-detection research because of relevant physical and
  chemical similarities.

### Why these samples are studied

In controlled research settings, TEP, DIMP and DMMP samples can support:

1. **Sensor development:** checking whether portable detection instruments can
   recognise a known simulant;
2. **Protective filtration studies:** evaluating adsorbents such as activated
   carbon used in protective equipment;
3. **Instrument calibration and training:** calibrating laboratory instruments
   such as chromatographs and spectrometers without exposing operators to real
   nerve agents.

The references below provide general chemical and defence-research background;
they are not claims that this repository has validated any field detector.

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

## Reproduce the preparation with Conda

```bash
conda create -n raman-preprocessing python=3.12 -y
conda activate raman-preprocessing
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

## Background references

1. [Diisopropyl methylphosphonate — ScienceDirect background](https://www.sciencedirect.com/topics/chemistry/diisopropyl-methylphosphonate)
2. [Organophosphorus simulants and detection research — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11861048/)
3. [Triethyl phosphate research background — Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/17518253.2024.2438068)
4. [DIMP/DMMP/TEP thermal-analysis context — ResearchGate](https://www.researchgate.net/figure/Evolution-of-a-t-max-DIMP-DMMP-8-DEMP-9-and-TEP-7-and-b-t-ign-DIMP-and_fig5_341081951)
5. [Dimethyl methylphosphonate — Wikipedia overview](https://en.wikipedia.org/wiki/Dimethyl_methylphosphonate)
