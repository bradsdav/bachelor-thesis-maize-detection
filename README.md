# Maize detection from incomplete Sentinel-2 time series

Code for my Bachelor's thesis (FEE CTU in Prague, 2022-2023).

The task is to tell whether a field is maize from its Sentinel-2 time series, using the
TimeSen2Crop dataset (Weikmann et al., *TimeSen2Crop: A Million Labeled Samples Dataset of
Sentinel 2 Image Time Series for Crop-Type Classification*, IEEE JSTARS 2021). The main question
is how well this works when the series is **incomplete**, for example part-way through the season
when the later observations don't exist yet. Each sample is 73 time steps × 10 channels, and the
classifier is a 1-D ResNet-50 written in PyTorch.

This is research code. Each script is one experiment, run as-is on the faculty GPU cluster, so
there is plenty of repetition and the iteration history is still visible (`conf1` → `conf4`,
`Graphs` → `Graphs_4`, `csv_gen` → `csv_gen_1`, …).

## What I tried

- **Baseline vs. robust model.** The baseline is trained on complete series. The robust model has
  a random-length end of every series zeroed out during training. Both are then fine-tuned with
  dropout.
- **Specialised models.** One model per fixed amount of missing data (7, 14, …, 70 of the 73 time
  steps), as a reference to compare the single robust model against.
- **Accuracy vs. completeness.** All of the above evaluated with 0–73 time steps hidden.
- **How far to trust a prediction.** Three confidence measures, compared using accuracy–coverage
  curves:
  - the softmax score (`C_s`)
  - the variance of Monte-Carlo dropout predictions (`C_d`)
  - a separate confidence network trained to predict whether the classifier is right (`C_cn`)
- **Side experiments.** The full 16-class crop classification, and a variant with 9 input
  channels instead of 10.

## Repository layout

| Folder | Contents |
| --- | --- |
| `src/` | Shared code: PyTorch datasets and the 1-D ResNet-50, with and without dropout |
| `data_prep/` | Building the datasets from raw TimeSen2Crop, class-balance statistics |
| `training/` | One script per training experiment |
| `evaluation/` | Testing trained models, dumping per-sample predictions to CSV |
| `analysis/` | Plots and confidence analysis on the saved results |
| `data/meta/` | Acquisition dates per tile and the 73-step time grid that `read.py` aligns everything to |
| `data/` | Datasets as pickled numpy arrays (not included, ~16 GB) |
| `weights/` | Trained model weights (not included) |
| `results/logs/` | Per-epoch training and validation accuracy and loss |
| `results/eval/` | Evaluation outputs that the analysis scripts read |
| `results/archive/` | Outputs of earlier runs, kept for reference (no script reads them) |

### File-name suffixes

The same suffixes appear in scripts, datasets, weights and results:

| Suffix | Meaning |
| --- | --- |
| `_or` | Main two-class dataset (maize / not maize) |
| `_hid`, `robust` | Trained with the end of the series randomly hidden |
| `_dr` | Fine-tuned with dropout |
| `_rand_hid` | Evaluated on randomly shortened series |
| `_16` | All 16 TimeSen2Crop classes |
| `_miss` | 9 input channels instead of 10 |
| `_pred`, `_conf` | Data and weights for the classifier and for the confidence network |
| `_7` … `_70` | Specialised model with that many time steps hidden |

## Running

Python 3.10:

```bash
pip install -r requirements.txt
```

Run scripts **from the repository root, as modules**. That way `src/` is importable and the
relative data paths resolve:

```bash
python -m analysis.Graphs_2             # runs straight away on the saved results
python -m training.train                # needs data/orig/
python -m evaluation.test_specialised   # needs data/orig/ and weights/
```

What runs without extra files:

- The plots built from text results (`Graphs`, `Graphs_2`, `Graphs_3`, `csv_1`) run without
  anything else.
- `conf1`–`conf4` need the prediction CSVs in `results/eval/`. These are left out of git because
  of their size (~230 MB); `evaluation/csv_gen*.py` regenerates them.
- Training and evaluation need the datasets in `data/`, and evaluation also needs the weights in
  `weights/`.

### Data

`data_prep/read.py` converts the raw TimeSen2Crop CSV files into numpy arrays aligned to a common
73-step time grid (the path to the raw dataset is set inside the script). The version here builds
the classifier and confidence-network training sets in `data/confidence/`; earlier versions of
the same script produced the other datasets:

- `data/orig/`: the main datasets, (train, val, test) split by regions (`*_or`, `*_16`, `*_miss`)
- `data/rand/`: the first version of the dataset, used by the earliest experiments, (train, val, test) split randomly by samples
- `data/confidence/`: separate training sets for the classifier (`*_pred`) and the confidence
  network (`*_conf`)
