# COMP30027 Project 2 — Team Notes

**Due:** Friday 22 May, 7:00 pm  
**Group size:** 2 → need 3–4 models per task, at least 1 ensemble in Task 2, report 2,500–3,000 words  
**Kaggle threshold to pass:** >50% accuracy on both tasks

---

## Project Structure

```
Machine-Learning---Project-2/
├── task1_data/          # ~4,000 train images, ~1,500 test images (64×64, 10 animal classes)
│   ├── images/train/
│   ├── images/test/
│   ├── train_metadata.csv    # columns: image_path, class_id, class_name
│   ├── test_metadata.csv     # columns: image_path (no labels)
│   ├── color_histogram.csv
│   ├── hog_pca.csv
│   └── additional_features.csv
└── task2_data/          # ~350–400 train images, ~150–200 test images (128×128, 10 bird species)
    └── (same structure as above)
```

---

## Phase 0 — Setup (DONE)
- [x] Git repo created, .gitignore set up (data excluded from git)
- [ ] Create a virtual environment and install packages
- [x] Create one notebook per task (`task1.ipynb`, `task2.ipynb`)

### Virtual Environment Setup (detailed steps)

A virtual environment is an isolated Python installation just for this project. It keeps your project's packages separate from other projects and your system Python, so nothing breaks each other.

**Step 1 — Open a terminal in the project folder**
- In VS Code: open the integrated terminal (`` Ctrl+` `` or Terminal menu → New Terminal)
- Make sure your terminal is inside `Machine-Learning---Project-2/` — you should see that in the prompt

**Step 2 — Create the virtual environment**
- Run the command to create a folder called `venv` inside your project
- This uses Python's built-in `venv` module — no extra install needed
- The command is: `python3 -m venv venv`
- This creates a `venv/` folder — it's already excluded from git by your `.gitignore`

**Step 3 — Activate the virtual environment**
- On Mac/Linux: `source venv/bin/activate`
- You'll know it worked when your terminal prompt shows `(venv)` at the start
- You need to do this every time you open a new terminal session

**Step 4 — Install the packages**
- Once activated, use `pip install` to install everything you need
- Install these one group at a time so it's easier to spot errors:
  1. Core data/ML tools: `pip install numpy pandas matplotlib seaborn scikit-learn`
  2. Image loading: `pip install Pillow opencv-python`
  3. Notebook support: `pip install ipykernel notebook`
  4. PyTorch (for ResNet feature extraction in Task 2):
     - Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
     - Select: Stable, Mac, Pip, Python, CPU (unless you have a GPU)
     - Copy and run the command it gives you — it includes `torchvision` automatically
  5. Extra feature tools: `pip install scikit-image` (for LBP texture features)

**Step 5 — Connect the virtual environment to your Jupyter notebook**
- After installing everything, run: `python -m ipykernel install --user --name=venv --display-name "Project 2 (venv)"`
- Then open your notebook in VS Code → click the kernel selector (top right) → choose "Project 2 (venv)"
- This makes sure your notebook uses the packages from your venv, not system Python

**Step 6 — Save your package list**
- Run `pip freeze > requirements.txt` to save a list of all installed packages
- Commit `requirements.txt` to git so your teammate can install the exact same versions
- Your teammate runs: `pip install -r requirements.txt` (after creating their own venv and activating it)

**Troubleshooting tips:**
- If `python3` is not found, try `python` instead
- If pip install fails for PyTorch, use the exact command from the pytorch.org website — do not guess it
- If VS Code doesn't show your venv as a kernel option, reload the window (Cmd+Shift+P → "Reload Window")

---

## Phase 1 — Data Loading & Exploration

### Step 1.1 — Load the CSVs
- Load `train_metadata.csv` → gives you image paths + labels
- Load `test_metadata.csv` → gives you image paths (no labels)
- Load the three feature CSVs for both train and test
- Check: how many rows each has, any missing values, class distribution (is it balanced?)

### Step 1.2 — Understand the features provided
- `color_histogram.csv`: binned colour counts across R, G, B channels — good for colour-based classes
- `hog_pca.csv`: shape/edge information, already reduced by PCA — good for structure
- `additional_features.csv`: edge density, texture variance, average channel values — compact summary stats
- Each CSV should have one row per image (train + test combined, or separate — check the row count)

### Step 1.3 — Visual exploration
- Display a few sample images from each class (use `matplotlib.pyplot.imshow`)
- Look for: obvious colour differences, shape differences, backgrounds
- For Task 2 specifically: notice which bird species look similar (Herring Gull vs Ring-Billed Gull, House Sparrow vs Song Sparrow, Yellow Warbler vs Wilson Warbler) — these will be your hard cases

**While looking at the images, write down answers to these questions (paste into the report later):**
- What makes each species visually distinct? (e.g. "Cardinal: all-red body, crest; Blue Jay: blue/white/black pattern")
- For each look-alike pair: what is the ONE visible difference, if any? (e.g. bill shape for the gulls)
- Are there background/lighting differences that might confuse a model even if the bird is the same?
- Do some images have the bird very small/off-centre? — these will cause errors regardless of model

**→ These observations go directly into the Discussion section of the report.**

### Step 1.4 — Feature exploration
- Scale the features first with `StandardScaler` (PCA needs this)
- Run PCA to reduce 219 features → 2 numbers, then plot coloured by class
  - **If classes form separate clusters** → provided features are decent, models will work okay
  - **If everything overlaps** → provided features are not enough, ResNet is essential
- Check for low-variance features (variance < 0.01) — these are nearly constant and won't help any model

---

## Phase 2 — Feature Engineering

**Goal: hit 95%+ accuracy on Task 2 (current leaderboard high is 95.55%)**

The provided features (colour histograms, HOG, edge stats) are too basic for fine-grained bird classification. The key to high accuracy is using a deep pretrained model as a feature extractor.

### Step 2.1 — Why ResNet works better than the provided features

The provided features describe images using hand-crafted rules:
- Colour histogram: counts how many pixels are each colour — misses WHERE the colour is
- HOG: captures edge directions — good for shape, bad for texture details
- Additional features: single numbers per image (edge density, avg colour) — very coarse

ResNet-50 was trained on 1.2 million ImageNet images and learned to detect thousands of visual patterns: textures, parts, shapes, colours in context. When you remove its last layer, the remaining 2048 numbers are a rich description of the image that captures fine details HOG and colour histograms simply can't.

### Step 2.2 — ResNet-50 feature extraction ✓ DONE

How it works:
1. Load ResNet-50 pretrained on ImageNet from `torchvision.models`
2. Remove the last layer (the 1000-class ImageNet classifier) — everything before it stays
3. Resize each image to 224×224 (what ResNet expects) and normalise with ImageNet mean/std
4. Pass each image through the model → get a vector of 2048 numbers
5. Save the result to CSV — extraction takes a few minutes, you don't want to repeat it

Saved files (already extracted):
- `task2_data/resnet_train.csv` — 417 rows × 2048 columns
- `task2_data/resnet_test.csv`  — 180 rows × 2048 columns

**Important rule:** Do NOT use any model pre-trained on CIFAR-10 or CUB-200-2011.

### Why ResNet? — write this in the report Methodology section

ResNet-50 (He et al., 2016) is a deep convolutional neural network trained on 1.28 million ImageNet images across 1,000 categories. Its key advantage as a feature extractor is that it has learned a rich, hierarchical representation of visual concepts:
- Early layers detect edges and colours
- Middle layers detect textures and patterns (feathers, stripes)
- Late layers detect object parts (beaks, wings, body shapes)

The provided features (colour histograms, HOG-PCA, edge density) are hand-crafted and capture only shallow statistics. They cannot distinguish, for example, a Herring Gull from a Ring-Billed Gull, because both birds have identical colour distributions and similar gradient patterns — the only difference is the banding on the bill, a fine detail that is invisible to colour histograms or HOG.

ResNet's 2048-dimensional output encodes all of these levels at once. When we pass two visually similar birds through ResNet, their feature vectors will be far apart in 2048-dim space if they differ in subtle texture or part details, even if their colour histograms are nearly identical.

Why not fine-tune ResNet directly?
- We only have ~42 images per class — not enough to update 25 million parameters without severe overfitting
- Using it as a frozen feature extractor (no gradient updates) avoids this problem entirely
- This is standard practice in transfer learning with small datasets (Donahue et al., 2014)

### Step 2.3 — Feature sets to use in models

| Name | What it is | Best for |
|------|-----------|----------|
| `X_train` | 219 provided features, combined | Baseline models |
| `X_train` (scaled) | Same but StandardScaler applied | SVM, kNN |
| ResNet features | 2048-dim from ResNet-50 | Main feature set for high accuracy |
| ResNet + provided (combined) | 2048 + 219 = 2267 features | Try this if ResNet alone isn't enough |

---

## Phase 3 — Models: Task 1 (teammate's task — for reference)

### Models to use (3–4 required):
1. **Baseline (DummyClassifier)** — always predicts the most common class. Floor that everything must beat.
2. **kNN** — try k = 3, 5, 7, 11 with `GridSearchCV`. Scale features first.
3. **SVM (RBF kernel)** — tune C and gamma with `GridSearchCV`. Best for medium-sized datasets.
4. **Random Forest** — 200 trees, no scaling needed. Try also as a 4th distinct model.

Evaluate all with **5-fold stratified CV** → report mean accuracy ± std.
Plot a confusion matrix for the best model.

---

## Phase 4 — Models: Task 2 (your task)

### What has been implemented in task2.ipynb:

**Step 3.1 — Load & scale ResNet features**
- Load `resnet_train.csv` and `resnet_test.csv` from `task2_data/`
- Apply `StandardScaler` — fit on train only, then transform both train and test with the same scaler
- This prevents data leakage (test data must never influence the scaler)

**Step 3.2 — Baseline**
- `DummyClassifier(strategy='most_frequent')` — ~10% accuracy (1 in 10 classes)
- Every real model must clearly beat this

**Step 3.3 — kNN on ResNet features**
- Try k = 3, 5, 7, 11 and pick the best from CV results
- kNN in ResNet space works because similar birds produce similar 2048-dim vectors
- Weakness: with only ~42 images per class, the nearest neighbours are not always reliable

**Step 3.4 — SVM (RBF kernel) on ResNet features**
- Tune C (strictness of boundary) and gamma (influence radius of each point)
- `GridSearchCV` tries all combinations over 5-fold CV automatically
- SVM with RBF kernel is the standard strong baseline for fixed feature vectors on small datasets
- Why it works: ResNet features lie in 2048-dim space; SVM finds the maximum-margin hyperplane that separates classes — even if classes overlap in 2D PCA, they may be linearly separable in 2048 dimensions

**Step 3.5 — Random Forest on ResNet features**
- 200 trees, no scaling needed (trees are invariant to feature scale)
- Generally weaker than SVM on ResNet features but useful in the ensemble

**Step 3.6 — Ensemble (VotingClassifier, soft voting) — REQUIRED**
- Combines SVM + kNN + RF
- Soft voting: each model outputs a probability per class → average the probabilities → pick highest
- Better than hard voting because confident predictions get more influence
- This is the required ensemble for groups of 2

**Step 3.7 — Confusion matrix**
- Run `cross_val_predict` to get CV predictions without re-fitting
- Plot with seaborn heatmap — look for which bird pairs are most confused

**Step 3.8 — Kaggle submission**
- Fit best model on ALL training data (not just one CV fold)
- Predict on test set
- Map class names → class IDs using `train_metadata.csv`
- Save as `task2_submission.csv` with columns `image_id`, `class_id`

### If accuracy is below 90% — next things to try:
- Try combining ResNet features + provided features (2048 + 219 = 2267 features)
- Try a stronger backbone: EfficientNet-B4 or ViT-B/16 from `torchvision`
  - EfficientNet: `models.efficientnet_b4(weights='IMAGENET1K_V1')`, remove last layer
  - ViT: `models.vit_b_16(weights='IMAGENET1K_V1')`, get the CLS token output
- Try Logistic Regression on ResNet features (sometimes beats SVM with the right C value)
- Try data augmentation: horizontal flip, small rotations — effectively increases training set size

---

## Phase 5 — Error Analysis (important for the report)

Do this for both tasks:

- Plot confusion matrix (use `seaborn.heatmap` on `sklearn.metrics.confusion_matrix`)
- Find the most confused class pairs — what do they have in common?
- Display some misclassified images side by side
- Ask: is the error due to the feature not capturing the difference, or is the model decision boundary wrong?

For Task 2 specifically, look at:
- Herring Gull vs Ring-Billed Gull (nearly identical appearance)
- House Sparrow vs Song Sparrow (both brown/grey)
- Yellow Warbler vs Wilson Warbler (both small yellow birds)

---

## Phase 6 — Kaggle Submission

**Current leaderboard high (Task 2): 95.55%** — this is the score to beat.

Submission format — two columns, no labels on test set:
```
image_id,class_id
test_0000,4
test_0001,7
test_0002,2
```

- `image_id` = the image ID from `test_metadata.csv` (e.g. `test_0000`)
- `class_id` = the **numeric** class ID (0–9), NOT the class name string
- Get the mapping from `train_metadata.csv` — columns `class_name` and `class_id`
- Run your best model on `X_test` → predicted class names → map to class IDs → save CSV → upload

Steps:
1. Train your best model on all of `X_train` (not just a CV fold — use everything)
2. Call `.predict(X_test)` → get predicted class names
3. Map class names to class IDs using the `class_id` column from `train_metadata.csv`
4. Build a DataFrame with columns `image_id` and `class_id`
5. Save as CSV with `index=False`
6. Upload to Kaggle (up to 8 submissions per day)

---

## Phase 7 — Full Report Outline (both tasks, 2,500–3,000 words)

> Bullet-point guide only — write in your own words. All numbers below are your actual results from the notebooks. Word counts are targets, not hard limits. Write this AFTER you have all results from both tasks.

---

### Actual Results Reference (both tasks)

**Task 1 — Coarse-grained animal classification**
- 3,750 train / 1,250 test / 64×64 px / 10 classes (perfectly balanced, 375 each)
- Provided features: 219 total (96 colour + 100 HOG-PCA + 23 additional)
- Engineered features: 12 pixel channel stats (mean + std per RGB channel) → 231 combined
- PCA: 19.9% variance explained in 2D
- Models (5-fold stratified CV):

| Model | Features | CV Accuracy |
|-------|----------|-------------|
| ZeroR | — | 10.0% ± 0.0% |
| kNN (k=11) | feat_combined | 26.9% ± 1.7% |
| SVM Linear (C=1) | feat_combined | 49.5% ± 1.3% |
| SVM RBF (C=10) | feat_combined | 52.5% ± 2.4% |
| SVM RBF (C=10) | ResNet-50 | 87.0% ± 1.3% |
| Random Forest (300 trees) | ResNet-50 | 86.7% ± 1.3% |
| **Logistic Regression** | **ResNet-50** | **90.0% ± 1.7%** |

- Error analysis (SVM RBF on provided features, best without ResNet):
  - Total misclassified: 1,780 / 3,750 (47.5%)
  - Worst classes: cat 36.8%, dog 40.8%, deer 43.7%
  - Best classes: butterfly 64.0%, horse 61.6%, elephant 59.5%
  - Top confused pairs: dog→cat (88×), cat→dog (79×), deer→bird (73×), spider→butterfly (58×), sheep→elephant (58×)
- Kaggle submission: Logistic Regression on ResNet-50 features

**Task 2 — Fine-grained bird species classification**
- 417 train / 180 test / 128×128 px / 10 species (~42 each, Cardinal: 39)
- Provided features: 219 total — 102/219 have near-zero variance (< 0.01)
- PCA: 17.0% variance explained in 2D (even worse than Task 1)
- ResNet-50 features: 2,048 dimensions
- Models (5-fold stratified CV):

| Model | Features | CV Accuracy |
|-------|----------|-------------|
| Baseline | — | 9.6% ± 0.1% |
| kNN (k=7) | ResNet-50 | 81.3% ± 2.5% |
| SVM RBF (C=10, γ=scale) | ResNet-50 | 85.1% |
| Random Forest (200 trees) | ResNet-50 | 86.1% ± 2.7% |
| **Ensemble (SVM+kNN+RF)** | **ResNet-50** | **86.8% ± 2.5%** |

- Kaggle test accuracy: **90.0%** (trained on full 417 images)

---


Use the provided LaTeX or Word template. Submit as a single PDF.
Write this section AFTER your teammate has their Task 1 results — you need both sets of numbers.

---

### Your actual Task 2 results (fill these into the report)

**Dataset facts:**
- Training: 417 images across 10 bird species (42 per class except Cardinal: 39)
- Test: 180 images (unlabelled)
- Provided features: 219 total (96 colour histogram + 100 HOG-PCA + 23 additional)
- ResNet-50 features: 2048 dimensions per image

**PCA finding (from Step 1.4):**
- PC1 explains only 11.5% of variance, PC2 explains 5.5% → only 17% total in 2D
- 102 out of 219 provided features have variance < 0.01 (nearly half are near-constant)
- This tells you the provided features are sparse and weak for this task

**Model results (5-fold stratified CV on training set):**

| Model | Features | CV Accuracy |
|-------|----------|-------------|
| Baseline (most frequent) | — | 9.6% ± 0.1% |
| kNN (k=7, best) | ResNet-50 | 81.3% ± 2.5% |
| kNN (k=5) | ResNet-50 | 81.1% ± 4.5% |
| SVM (RBF, C=10, γ=scale) | ResNet-50 | 85.1% |
| Random Forest (200 trees) | ResNet-50 | 86.1% ± 2.7% |
| **Ensemble (SVM+kNN+RF, soft voting)** | **ResNet-50** | **86.8% ± 2.5%** |

**Kaggle test accuracy: 90.0%** (trained on all 417 images, tested on 180)

Note: Kaggle test accuracy (90%) is higher than CV accuracy (86.8%) because the final model is trained on the full training set instead of 4/5 of it — more data → better generalisation.

---

### Section 1 — Introduction (~250 words, write this LAST)

**Task description (~100 words):**
- Task 1: classify 10 animal categories (bird, butterfly, cat, deer, dog, elephant, frog, horse, sheep, spider) — 3,750 training images, 64×64 px, from CIFAR-10 and public datasets (cite: Krizhevsky, 2009)
- Task 2: classify 10 bird species from CUB-200-2011 (cite: Wah et al., 2011) — 417 training images, 128×128 px
- The coarse-to-fine theme: Task 1 classes are visually distinct; Task 2 classes share the same body plan and differ only in subtle details

**Why it matters (~50 words):**
- Fine-grained recognition is significantly harder — standard features that work for coarse tasks fail here
- This mirrors real-world challenges in medical imaging, wildlife monitoring, and product identification

**Your approach (~100 words):**
- Both tasks: used three provided feature sets (colour histogram, HOG-PCA, additional stats) plus engineered ResNet-50 deep features
- Task 1: evaluated ZeroR, kNN, SVM (RBF and linear), then ResNet-50 → best was Logistic Regression on ResNet (90.0% CV)
- Task 2: evaluated baseline, kNN, SVM, Random Forest, ensemble → best was soft-voting ensemble on ResNet (86.8% CV, 90.0% Kaggle)
- State these are the results; the report explores why

---

### Section 2 — Methodology (~650 words total)

**2.1 — Features (~200 words, covers both tasks)**
- Three feature sets were provided for both tasks:
  - Colour histogram: 96 dims — bins pixel intensities across R, G, B channels; captures colour distribution but not spatial location
  - HOG-PCA: 100 dims — gradient-based features capturing edge direction and shape structure, reduced by PCA
  - Additional stats: 23 dims — edge density, texture variance, average colour per channel
  - Total: 219 provided dimensions per image
- All three concatenated into one matrix (feat_basic)
- StandardScaler applied (zero mean, unit variance) before SVM and kNN — distance-based models require this
- Additional engineered features (Task 1): mean and standard deviation per RGB channel extracted from raw pixels → 12 extra features → 231-dim feat_combined
- ResNet-50 used as a deep feature extractor for both tasks:
  - Pretrained on ImageNet (1.28M images, 1,000 categories) — cite He et al., 2016
  - Final classification layer removed; each image passed through → 2,048-dim feature vector
  - Images resized to 224×224, normalised with ImageNet mean/std before extraction
  - Allowed because ResNet was not trained on CIFAR-10 or CUB-200-2011
  - Motivation: PCA on provided features retains only 19.9% (Task 1) and 17.0% (Task 2) of variance in 2D — poor class separability

**2.2 — Models (~250 words, covers both tasks)**

Write one short paragraph per model. Mention which tasks each was used for.

- **ZeroR (Task 1 only):** always predicts the most frequent class regardless of input — 10% accuracy; serves as the floor every real model must beat
- **kNN:** classifies by majority vote among the k nearest training points in feature space; k tested over {1, 5, 11, 21} for Task 1 and {3, 5, 7, 11} for Task 2; sensitive to feature scale so StandardScaler applied first
- **SVM with RBF kernel:** finds the maximum-margin hyperplane between classes; C controls regularisation strength, γ controls influence radius per training point; both tuned by grid search; used on feat_basic, feat_combined, and ResNet features
- **SVM with Linear kernel (Task 1 only):** straight-line decision boundary; compared against RBF to understand whether non-linearity helps
- **Random Forest (Task 1 and 2):** ensemble of 300/200 decision trees trained on random feature subsets; votes averaged; no scaling needed; robust to irrelevant features
- **Logistic Regression (Task 1 only):** linear model with probability output; particularly effective on high-dimensional ResNet features
- **Soft-voting Ensemble (Task 2):** combines SVM + kNN + RF by averaging their predicted class probabilities; required for groups of 2

**2.3 — Evaluation (~100 words)**
- Stratified 5-fold cross-validation on the labelled training set for all models
- Stratified = each fold preserves class proportions — critical for Task 2 with only ~42 images per class
- Primary metric: classification accuracy (matches Kaggle scoring)
- Final model retrained on the full training set before generating test predictions
- Kaggle competitions used as external validation on held-out test data

---

### Section 3 — Results (~400 words total)

**3.1 — Task 1 results (~200 words)**

Include results table:

| Model | Features | CV Accuracy |
|-------|----------|-------------|
| ZeroR | — | 10.0% ± 0.0% |
| kNN (k=11) | feat_combined | 26.9% ± 1.7% |
| SVM Linear (C=1) | feat_combined | 49.5% ± 1.3% |
| SVM RBF (C=10) | feat_combined | 52.5% ± 2.4% |
| SVM RBF (C=10) | ResNet-50 | 87.0% ± 1.3% |
| Random Forest (300 trees) | ResNet-50 | 86.7% ± 1.3% |
| **Logistic Regression** | **ResNet-50** | **90.0% ± 1.7%** |

- Reference the table in text
- Best provided-features model: SVM RBF (52.5%)
- Best overall: Logistic Regression on ResNet (90.0% CV)
- Kaggle test accuracy: [fill in from Kaggle after your teammate submits]
- Include confusion matrix as a figure — caption it, reference it in text
- Note the most confused pairs from the confusion matrix: dog↔cat (88+79 times), deer→bird (73 times), spider↔butterfly (58 times), sheep↔elephant (58 times)

**3.2 — Task 2 results (~200 words)**

Include results table (from results reference above).

- Reference the table in text
- Best model: ensemble on ResNet (86.8% CV), 90.0% Kaggle
- Compare to baseline (9.6%) to show how much was gained
- Note: Kaggle score (90%) > CV score (86.8%) — because final model uses all 417 images
- Include Task 2 confusion matrix as a figure — reference it
- Point out the most confused pairs from the matrix

---

### Section 4 — Discussion (~900 words total, most important section)

**A — Why provided features worked for Task 1 but failed for Task 2 (~200 words)**
- Task 1 with provided features: SVM achieved 52.5% — well above baseline (10%) and above the 50% Kaggle threshold
- Task 2 with provided features: would be near-random — 102/219 features are near-zero variance, PCA retains only 17% in 2D
- Why features work for Task 1: animal categories are visually distinct — a frog and a horse have completely different colours, shapes, and textures; colour histograms alone give strong signal
- Why they fail for Task 2: all 10 birds are similar in colour range (browns, greys, yellows, whites) and share the same body shape; the discriminating information is in fine local details like bill banding or breast spots
- Theoretical link: this is a bias problem, not a variance problem — even a perfect classifier cannot separate the classes if the features don't carry the right information
- Evidence: switching to ResNet features in Task 1 jumped SVM from 52.5% → 87.0%, and in Task 2 from near-random → 85.1%

**B — Model comparison across both tasks (~200 words)**
- Task 1: kNN performed poorly even at best k=11 (26.9%) — with 219 provided features, the distance metric is noisy (curse of dimensionality); many irrelevant features dilute the signal
- Task 1: SVM RBF beat SVM Linear (52.5% vs 49.5%) → non-linear decision boundaries matter even for coarse classes; some animal pairs (cat/dog, bird/butterfly) overlap in feature space
- Task 1 with ResNet: Logistic Regression (90.0%) beat SVM RBF (87.0%) — in 2048-dim ResNet space, a linear classifier suffices because ResNet already learned non-linear transformations; adding more non-linearity (RBF kernel) gives no benefit and may slightly overfit
- Task 2: kNN improved dramatically on ResNet (81.3%) vs provided features — ResNet embeddings make the nearest-neighbour measure meaningful; similar species produce similar 2048-dim vectors
- Task 2: RF (86.1%) was competitive with SVM (85.1%) — both work well on ResNet features
- Task 2: ensemble (86.8%) improved slightly — each model errors differently, soft voting averages out some mistakes
- Key finding: model choice mattered less than feature choice in both tasks

**C — Error analysis (~250 words)**

Task 1 errors:
- Most confused pair: dog↔cat (88+79 times) — both quadrupeds with similar fur textures and body shapes; colour histograms cannot distinguish them
- Second worst: deer→bird (73 times) — deer images likely had backgrounds (sky, trees) that matched the colour profile of bird images; HOG features probably captured similar elongated shapes
- spider↔butterfly (58 times) — both have thin legs and detailed textures; at 64×64 the fine structure is lost
- sheep↔elephant (58 times) — both are large grey/white animals; similar colour distributions
- Best-classified: butterfly (64.0%), horse (61.6%) — distinctive shapes and colour patterns survive feature extraction even at coarse scale
- Worst-classified: cat (36.8%), dog (40.8%) — inter-class similarity is the bottleneck

Task 2 errors (read your confusion matrix for the specific numbers):
- Expected worst pairs: Herring Gull / Ring-Billed Gull — identical grey/white colouring; only difference is a narrow bill ring invisible at 128×128
- House Sparrow / Song Sparrow — both small brown streaked birds; Song Sparrow's breast spot is subtle and pose-dependent
- Yellow Warbler / Wilson Warbler — both yellow; Wilson's black cap IS distinctive, so confusion should be lower
- Well-classified: Cardinal (all-red, unique), American Goldfinch (bright yellow + black wings)
- The pattern: all major errors occur between visually similar pairs — errors are not random, they are predictable from visual inspection

**D — Coarse-to-fine reflection (~150 words)**
- The central lesson: as inter-class similarity increases, the feature representation becomes the bottleneck — not the model
- In Task 1, switching models (kNN → SVM) gained ~26 percentage points on provided features. In Task 2, switching models gained only ~5 points (kNN to ensemble on ResNet)
- In both tasks, switching from provided features to ResNet gained ~35–40 percentage points — far more than any model change
- Two compounding difficulties in Task 2: (1) 10× fewer training images, (2) 10× higher visual similarity between classes
- Theoretical link — bias vs variance: provided features have high representational bias for fine-grained tasks; ResNet reduces this bias. With fewer images, simpler models (Logistic Regression, SVM) control variance better than complex ones
- Real-world implication: for any fine-grained recognition task, invest in features first, then tune models

---

### Section 5 — Conclusion (~200 words, write LAST)

- One paragraph — summarise both tasks together, do not repeat results in detail
- Cover these points:
  - Both tasks: provided features were reasonable for Task 1 (52.5%), insufficient for Task 2
  - ResNet-50 transfer features dramatically improved both tasks (Task 1: 90.0%, Task 2: 90.0% Kaggle)
  - The progression from coarse to fine-grained revealed that feature quality is the dominant factor when classes become visually similar
  - Model diversity helped modestly in Task 2 (ensemble vs single model)
  - Future work: fine-tuning ResNet on the target dataset with data augmentation would likely push Task 2 beyond 90%; a stronger backbone (EfficientNet, ViT) may help further
  - Broader takeaway: fine-grained recognition is an open problem — even state-of-the-art models struggle with within-species variation and small training sets

---

### Section 6 — References

- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. *Technical Report*.
- Wah, C., et al. (2011). The Caltech-UCSD Birds-200-2011 Dataset. *CNS-TR-2011-001*.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. *CVPR*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

---

### Writing tips
- No code or variable names in the report — describe concepts, not syntax
- Every table and figure must be referenced in the text before it appears
- Methodology = what you did and why you chose it. Results = what happened. Discussion = why
- Write Discussion as you go — after each experiment, note one sentence explaining the result
- Introduction and Conclusion are written last
- Teammate writes Task 1 portions of each section; you write Task 2 portions — combine into one document

---

## Division of Work (suggested)

| Task | Person |
|------|--------|
| Data loading & exploration | Both |
| Feature engineering (basic) | Person A |
| Feature engineering (ResNet) | Person B |
| Task 1 models | Person A |
| Task 2 models | Person B |
| Error analysis | Both |
| Report writing | Both |

---

## Key Reminders
- Models pre-trained on CIFAR-10 or CUB-200-2011 are **not allowed** as feature extractors
- ImageNet pre-trained models (ResNet, VGG, etc.) **are allowed**
- Must submit predictions to **both** Kaggle competitions
- Group of 2 needs **at least 1 ensemble** model in Task 2
- Report must be submitted as **PDF**
- Code must be runnable and support your reported results
