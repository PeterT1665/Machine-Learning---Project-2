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

### Step 1.4 — Feature exploration
- Plot the distribution of a few feature columns (histograms)
- Try a quick PCA visualisation of the combined features coloured by class — do classes cluster?
- Check for features with very small variance (may not be useful)

---

## Phase 2 — Feature Engineering

### Step 2.1 — Combine and preprocess provided features
- Concatenate the three feature CSVs into one feature matrix per image
- Apply `StandardScaler` (zero mean, unit variance) — important for SVM and kNN
- Optional: apply PCA again on the combined features to reduce dimensionality further

### Step 2.2 — Engineer additional features from raw images
You must go beyond the provided features. Ideas in order of effort:

**Low effort (do these first):**
- Flatten the raw pixel values as a feature vector (very high-dimensional but simple baseline)
- Compute mean/std per colour channel per image

**Medium effort:**
- Extract HOG features yourself with different parameters than what's provided (different cell size, orientations)
- Compute LBP (Local Binary Patterns) for texture — good for fine-grained Task 2
- Compute colour statistics in HSV space (not just RGB)

**Higher effort (recommended for Task 2):**
- Use a pre-trained ImageNet model (e.g. ResNet-50 from torchvision) as a feature extractor
  - Load the model, remove the final classification layer
  - Pass each image through → get a feature vector (e.g. 2048-dim for ResNet-50)
  - This is allowed because ResNet was NOT trained on CIFAR-10 or CUB-200-2011
  - This will likely be your strongest feature set for Task 2

**Important rule:** Do NOT use any model pre-trained on CIFAR-10 or CUB-200-2011.

### Step 2.3 — Build your final feature sets
Create a few named feature combinations to experiment with, e.g.:
- `feat_basic` = provided features only (combined + scaled)
- `feat_pixels` = raw pixel values (scaled)
- `feat_combined` = provided features + your engineered features
- `feat_resnet` = ResNet-50 embeddings (Task 2 mainly)

---

## Phase 3 — Models: Task 1 (need 3–4 distinct models)

### Suggested models (pick 3–4):

**Model 1 — Baseline (ZeroR or simple kNN)**
- ZeroR = always predict the majority class. Use this as your floor — anything below this is broken.
- Or 1-NN as a slightly stronger baseline

**Model 2 — k-Nearest Neighbours (kNN)**
- Try k = 1, 5, 11, etc. using cross-validation to find the best k
- Use `StandardScaler` before fitting — kNN is distance-based, scale matters
- Feature set: try `feat_basic` first, then `feat_combined`

**Model 3 — Support Vector Machine (SVM)**
- Try RBF kernel and linear kernel — these count as two distinct models if performance differs significantly
- Use `GridSearchCV` to tune C and gamma
- SVM is usually strong on medium-sized datasets with good features
- Feature set: `feat_basic` and `feat_combined`

**Model 4 — Random Forest**
- Start with 100–200 trees, try tuning `max_depth` and `min_samples_leaf`
- Does not need scaling
- Feature set: try all feature sets

**Optional extras:**
- Logistic Regression (strong linear baseline)
- Gradient Boosting (XGBoost or sklearn's GradientBoostingClassifier)

### Evaluation approach for Task 1:
- Use **5-fold cross-validation** on the training set to compare models
- Report: mean accuracy + std across folds
- Use a **confusion matrix** to see which classes are confused
- Pick the best model → use it to predict the test set → submit to Kaggle

---

## Phase 4 — Models: Task 2 (need 3–4 distinct models, at least 1 ensemble)

Task 2 is harder (fewer images, visually similar classes). Apply what worked in Task 1, but expect worse results.

### Step 4.1 — Apply best Task 1 models
- Re-run your best Task 1 models on Task 2 data
- Compare accuracy — it will likely drop significantly
- This comparison is important material for the Discussion section of your report

### Step 4.2 — Feature matters more here
- The provided features are likely not enough for fine-grained classification
- ResNet-50 embeddings will help a lot here — implement this before trying more models

### Suggested models for Task 2:

**Model 1 — SVM on ResNet features**
- Extract ResNet-50 features, apply SVM (RBF kernel)
- This combination tends to work well on small fine-grained datasets

**Model 2 — kNN on ResNet features**
- Simple but often competitive when features are good

**Model 3 — Random Forest on combined features**

**Model 4 (REQUIRED for groups of 2) — Ensemble**
- **Voting classifier:** combine predictions from SVM + Random Forest + kNN via majority vote
  - Use `sklearn.ensemble.VotingClassifier` with `voting='soft'` (uses predicted probabilities)
- OR **Stacking:** train a meta-classifier on the outputs of your base models
  - Use `sklearn.ensemble.StackingClassifier`
  - Base models: SVM, RF, kNN
  - Meta-classifier: Logistic Regression (simple and interpretable)

### Evaluation approach for Task 2:
- Dataset is small (~350–400 train), so use **stratified 5-fold CV** (makes sure each fold has all classes)
- Same confusion matrix analysis — pay attention to the similar-looking bird pairs
- Submit best model predictions to Kaggle

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

Format for submission (check the Kaggle example file):
- Likely: CSV with two columns — `id` (image filename or ID) and `label` (predicted class name)
- Run your best model on the test set, save predictions as CSV, upload to Kaggle

---

## Phase 7 — Report (2,500–3,000 words for a group of 2)

### Sections:

1. **Introduction** (~200 words)
   - Describe both tasks, the coarse-to-fine theme, brief summary of what you did

2. **Methodology** (~600–700 words)
   - Features: what you used and why (provided features, engineered features, ResNet)
   - Models: what you chose and why (conceptual level, not code)
   - Evaluation: how you validated (CV strategy, metric used)

3. **Results** (~300–400 words)
   - Table of model accuracies for Task 1
   - Table of model accuracies for Task 2
   - Confusion matrices (include as figures)

4. **Discussion and Critical Analysis** (~700–800 words) ← most important
   - Why does Task 2 have lower accuracy than Task 1?
   - Which features help most for each task and why?
   - Which classes are most confused and what does this tell us?
   - Connect to theory: bias-variance, the curse of dimensionality, decision boundaries
   - What does the coarse-to-fine progression reveal about ML limitations?

5. **Conclusion** (~150 words)

6. **References**

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
