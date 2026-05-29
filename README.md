# COMP30027 Machine Learning — Project 2

Two image classification tasks using scikit-learn and PyTorch.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=venv --display-name "Project 2 (venv)"
```

Open each notebook in VS Code and select the **"Project 2 (venv)"** kernel before running.

---

## Task 1 — Coarse-Grained Animal Classification (`task1.ipynb`)

10 animal classes, 3,750 train / 1,250 test images (64×64 px).

### How to run

Open `task1.ipynb` and run all cells top to bottom (`Run All`). Expected total runtime: ~15–20 minutes (most time is ResNet feature extraction in Section 11).

### Cell-by-cell guide

| Section | Cells | What it does |
|---------|-------|--------------|
| 1–2 | Import & Load data | Loads CSVs from `task1_data/` |
| 3–4 | Data quality & label checks | Verifies no missing values, prints class distribution |
| 5 | Sample images | Displays one image per class |
| 6 | Feature exploration | Summary stats and variance check on provided features |
| 7.1 | `feat_basic` (219-dim) | Concatenates and scales provided features |
| 7.2 | `feat_combined` (231-dim) | Adds 12 pixel/HSV channel stats to feat_basic |
| 7.3 | PCA visualisation | 2D scatter plot of training set by class |
| 8.1 | **ZeroR baseline** | 10.0% ± 0.0% — majority-class floor |
| 8.2 | **kNN (k=1,5,11,21)** | Best: k=11 on feat_combined → 26.9% ± 1.7% |
| 8.3 | **SVM RBF (C=1,10)** | Best: C=10 on feat_combined → 52.5% ± 2.4% |
| 8.4 | **SVM Linear (C=1)** | Best: feat_combined → 49.5% ± 1.4% |
| 8.5 | Results summary table | Ranked table of all CV results |
| 9.1 | Confusion matrix | SVM RBF C=10 on feat_combined via 5-fold CV |
| 9.2 | Per-class accuracy | Worst→best breakdown + top confused pairs |
| 9.3 | Misclassified images | Grid of 18 misclassified examples |
| 10 | First Kaggle submission | Trains SVM RBF on all 3,750 images → saves `task1_submission.csv` |
| 11 | ResNet-50 extraction | Extracts 2,048-dim features for all 5,000 images (~10 min) |
| 11.1 | **Models on ResNet** | SVM RBF → 87.0%, **LogReg → 90.0%**, RF → 86.7% |
| 11.2 | Best Kaggle submission | Trains best model on full set → overwrites `task1_submission.csv` |

### Where to find results

| Result | Location |
|--------|----------|
| All CV accuracy scores | Printed output of cells in Section 8 and 11.1 |
| Ranked results table | Output of Section 8.5 |
| Confusion matrix figure | Output of Section 9.1 |
| Per-class accuracy table | Output of Section 9.2 |
| Kaggle submission file | `task1_submission.csv` (project root) |

---

## Task 2 — Fine-Grained Bird Species Classification (`task2.ipynb`)

10 bird species, 417 train / 180 test images (128×128 px).

### How to run

Open `task2.ipynb` and run all cells top to bottom. Expected total runtime: ~5–10 minutes (ResNet features are pre-saved; only need re-extraction if `task2_data/resnet_train.csv` is missing).

### Cell-by-cell guide

| Section | What it does |
|---------|--------------|
| Load CSVs | Loads `train_metadata.csv`, `test_metadata.csv`, and the 3 feature CSVs from `task2_data/` |
| Data quality checks | Verifies row counts and missing values |
| Visual exploration | 3 samples per class; side-by-side hard look-alike pairs (Gull/Gull, Sparrow/Sparrow, Warbler/Warbler) |
| Feature exploration (PCA) | PCA scatter plot — confirms 17% variance in 2D; flags 102 near-zero-variance features |
| ResNet extraction | Loads ResNet-50 (ImageNet weights), extracts 2,048-dim features, saves `task2_data/resnet_train.csv` and `task2_data/resnet_test.csv`. **Skipped automatically if CSVs already exist — re-run this section only if data is missing.** |
| ResNet PCA | PCA scatter of ResNet features — compare clustering vs provided features |
| Load & scale ResNet | Loads saved CSVs, applies `StandardScaler` (fit on train only) |
| **Baseline** | DummyClassifier → 9.6% ± 0.1% |
| **kNN (k=3,5,7,11)** | Best: k=7 → 81.3% ± 2.5% |
| **SVM RBF (GridSearch)** | Best params: C=10, γ=scale → 85.1% CV |
| **Random Forest (200 trees)** | 86.1% ± 2.7% |
| **Ensemble (SVM+kNN+RF)** | Soft-voting VotingClassifier → **86.8% ± 2.5%** |
| Confusion matrix | Cross-validated predictions on SVM best model |
| Kaggle submission | Trains best model on all 417 images → saves `task2_submission.csv` |

### Where to find results

| Result | Location |
|--------|----------|
| All CV accuracy scores | Printed output of each model cell |
| Confusion matrix figure | Output of confusion matrix cell |
| Kaggle submission file | `task2_submission.csv` (project root) |
| Pre-extracted ResNet features | `task2_data/resnet_train.csv`, `task2_data/resnet_test.csv` |

---

## Data Layout

Data directories are not tracked in git and must be present locally:

```
task1_data/
  images/train/<class>/
  images/test/
  train_metadata.csv
  test_metadata.csv
  color_histogram.csv
  hog_pca.csv
  additional_features.csv

task2_data/
  images/train/<species>/
  images/test/
  train_metadata.csv
  test_metadata.csv
  color_histogram.csv
  hog_pca.csv
  additional_features.csv
  resnet_train.csv       # pre-extracted, generated by task2.ipynb
  resnet_test.csv        # pre-extracted, generated by task2.ipynb
```

---

## Key Results Summary

| Task | Model | Features | CV Accuracy | Kaggle |
|------|-------|----------|-------------|--------|
| Task 1 | Logistic Regression | ResNet-50 (2048-dim) | 90.0% ± 1.7% | — |
| Task 1 | SVM RBF (C=10) | ResNet-50 (2048-dim) | 87.0% ± 1.3% | — |
| Task 1 | SVM RBF (C=10) | Combined (231-dim) | 52.5% ± 2.4% | — |
| Task 2 | Ensemble (SVM+kNN+RF) | ResNet-50 (2048-dim) | 86.8% ± 2.5% | 90.0% |
| Task 2 | Random Forest (200 trees) | ResNet-50 (2048-dim) | 86.1% ± 2.7% | — |
| Task 2 | SVM RBF (C=10) | ResNet-50 (2048-dim) | 85.1% | — |
