# Shallow-and-Deep-Learning-Model-Comparison

# Neural Networks — Adult (UCI) & MNIST Experiments

**Shallow vs Deep models, vanishing-gradient analysis, reproducible notebooks & results**

---

## Project objective

This project investigates how **network depth** affects performance for two different data modalities:

* **Adult (UCI)** — tabular classification (predict `>50K` income).
* **MNIST** — image classification (handwritten digits 0–9).

We test **shallow** (1–3 hidden layers) vs **deep** (≥8 layers) models, measure classification performance, and demonstrate optimization issues (specifically **vanishing gradients**) and remedies. The output includes runnable notebooks, reusable data loaders, model definitions, training & evaluation scripts, and a concise 2–4 page report.

---

## High-level process

1. **Data acquisition & preprocessing**

   * Adult: clean missing values (`?`), encode categories, scale numerics.
   * MNIST: normalize pixel values, add channel / flatten depending on model.
2. **Baseline models**

   * Logistic regression / simple MLPs (shallow).
3. **Deep models**

   * Heavier MLPs / deeper convnets for MNIST.
4. **Diagnostics**

   * Training & validation curves; gradient-norm extraction per layer (to show vanishing gradients).
5. **Evaluation**

   * Accuracy, precision/recall, ROC-AUC (Adult), confusion matrix (MNIST).
6. **Report**

   * Figures + explanation of observed phenomena and recommendations.

---


## Data — what goes where (detailed)

* `data/raw/adult.data`

  * Put the original UCI Adult raw file here (download from: [https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)).
  * This is the canonical source file (do **not** commit very large generated binary files or processed data if your repo is public — prefer pointers and `.gitignore` for heavy files).

* `data/raw/mnist.npz`

  * Optional: you can include a copy of TensorFlow’s `mnist.npz` if you prefer offline usage. Typically unnecessary because TF can download it at runtime.

* `data/processed/adult_train.npz` (suggested contents)
  Save using `np.savez_compressed()` with keys:

  ```
  X_train = <numpy array after preprocessing>      # shape (n_train, n_features)
  y_train = <numpy array (n_train,) >
  X_val = ...
  y_val = ...
  X_test = ...
  y_test = ...
  preprocessor = ... (optional - you can save means/stds or sklearn pipeline)
  ```

  This makes training reproducible without re-running preprocessing steps.

* `data/processed/mnist_preprocessed.npz`

  * store `(X_train, y_train, X_val, y_val, X_test, y_test)` as float32 / categorical as needed (e.g. one-hot encoded `y_train_cat`).

**Example: save .npz in Python**

```python
import numpy as np
np.savez_compressed('adult_train.npz',
                    X_train=X_train_enc, y_train=y_train,
                    X_val=X_val_enc, y_val=y_val,
                    X_test=X_test_enc, y_test=y_test)
```

> **Note:** In `data/raw/README.md` include links & exact commands to download original datasets and a short checksum if desired.

---

## Procedure — step-by-step (how to reproduce)

### 1. Setup environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` should contain:

```
numpy
pandas
scikit-learn
matplotlib
tensorflow>=2.10
python-dotenv (optional)
```

### 2. Prepare data

* Put `adult.data` into `data/raw/` or run the notebook `notebooks/01_data_prep_adult.ipynb` which:

  * downloads `adult.data` if missing,
  * cleans missing values,
  * fits the preprocessing pipeline on train,
  * saves processed arrays to `data/processed/adult_train.npz`.

* Run `notebooks/03_data_prep_mnist.ipynb` or use TF loader to prepare `mnist_preprocessed.npz`.

### 3. Train models

* Use `notebooks/02_adult_models.ipynb` for Adult shallow & deep models (or run `src/train.py --config configs/adult_shallow.yaml` if you use CLI).
* Use `notebooks/04_mnist_models.ipynb` for MNIST.

### 4. Evaluate

* Run `src/evaluate.py` (or cells in notebooks) to generate:

  * test metrics (accuracy, precision/recall, AUC for Adult),
  * confusion matrix (MNIST),
  * save plots to `results/plots/`.

### 5. Generate report

* `docs/report.md` contains the narrative, tables and pointers to the PNGs in `results/plots/`.

---

## Results summary (what we observed — brief & actionable)

> **Adult (tabular)**

* Shallow model (2 hidden layers) — **Test Accuracy ≈ 0.781**
* Deep MLP (8 layers) — **Test Accuracy ≈ 0.767** (overfitting observed)
* **Vanishing gradients** confirmed by gradient-norm measurement on 10-layer sigmoid model: early layers near zero while later layers large → optimization difficulty.

> **MNIST (images)**

* Shallow fully-connected baseline — **Test Accuracy ≈ 0.977**
* Deep fully-connected / CNN — **Test Accuracy ≈ 0.9828**
* For MNIST, deeper models **improve** performance; vanishing gradient not a practical issue with ReLU + BatchNorm; conv architectures exploit spatial structure.

(Include a table `results/metrics.csv` with all experiment runs: model_id, dataset, architecture, hyperparams, train_acc, val_acc, test_acc, notes.)

---


**Where to place the plots in the narrative**

* Immediately after the paragraph describing the model & hyperparams for that experiment.
* Add figure captions: what model, dataset split, seed and a short interpretation.

**Note for graphs (.png) generation**

* Use Keras `History` objects to generate training/validation curves. Save with `plt.savefig("results/plots/Adult_shallow_accuracy.png", dpi=300)`.
* For gradient norms: generate a per-layer vector and plot with `plt.yscale('log')`.

---

## Reproducibility & runtime notes

* Use consistent seeds:

```python
import numpy as np, tensorflow as tf, random, os
SEED=42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)
```

* For large runs or faster training use Google Colab (GPU) — notebooks include a “Runtime → Change runtime type → GPU” reminder.
* Save model weights (`model.save('results/models/adult_shallow.h5')`) and the preprocessing pipeline (pickle the sklearn pipeline or store scalers/encoder mapping).

---

## How to push this project to GitHub (recommended workflow)

1. Initialize local repo (if not already):

```bash
git init
git add .
git commit -m "Initial commit — project structure and notebooks"
```

2. Create remote repository on GitHub (via web UI).
3. Add remote & push:

```bash
git remote add origin git@github.com:youruser/repo-name.git
git branch -M main
git push -u origin main
```

4. Large files (models, datasets): do **not** commit big binaries; use `.gitignore` and list them in `data/README.md`. For heavy files consider Git LFS.

**Suggested .gitignore**

```
__pycache__/
*.pyc
.env
venv/
*.h5
*.npz
/results/models/
data/processed/
```

---



