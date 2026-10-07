# bus-fusion-ga-eval

Code for a fold-consistent evaluation of mask-guided image fusion and genetic-algorithm (GA)
feature selection for three-class breast ultrasound classification (benign, malignant, normal).

The pipeline fuses each image with a lesion mask, extracts 512-d features from a CNN with a frozen
MobileNetV1 backbone, selects features with a GA, and classifies with a weighted soft-voting
ensemble (DT, KNN, RF, GB). Four input conditions are compared:

| Condition | Mask fused with the image |
|---|---|
| Raw | none |
| Predicted | U-Net mask trained inside the outer fold |
| Shuffled | expert mask of another image of the same class |
| Oracle | expert mask of the image |

All steps that learn from data are fitted on the outer-training images of each fold only.

## Data

- **BUSI** (Al-Dhabyani et al., *Data in Brief*, 2020), 780 images.
- **BUS-UCLM** (Vallez et al., *Scientific Data*, 2025), 683 images from 38 patients.

Both datasets are public and are not redistributed here. Place them as follows:

```
data/
├── BUSI/
│   ├── benign/       benign (1).png, benign (1)_mask.png, ...
│   ├── malignant/
│   └── normal/
└── BUS-UCLM/
    ├── images/
    └── masks/
```

## Usage

```
pip install -r requirements.txt
```

Run the notebooks in order, once per dataset (set `DATASET` at the top of each notebook):

| Notebook | Content | Hardware |
|---|---|---|
| `1_pipeline.ipynb` | outer folds, input conditions, predicted masks, features | GPU |
| `2_feature_selection.ipynb` | six GA configurations and feature-selection baselines | CPU, many cores |
| `3_evaluation.ipynb` | pooled results, bootstrap CI, McNemar, Friedman, Wilcoxon, stability | CPU |

Intermediate files are written to `outputs/<dataset>/`. The GA runs dominate the runtime.

## Results

`results/` contains the main values reported in the paper:

- `pooled.csv`: pooled out-of-fold macro-F1 with 95% bootstrap CI and per-class recall (standard GA, seed 0)
- `mcnemar.csv`: exact McNemar tests between conditions
- `baselines.csv`: standard GA versus all features, ANOVA, mutual information and random subsets of the same size

## Environment

Python 3.13, TensorFlow 2.20, scikit-learn 1.6.1, PyWavelets 1.8.0.
