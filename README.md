# Sentiment Analysis of Mental Health Social Media Text (Corrected)

This repository is a structurally and methodologically corrected version of the project: [Sentiment-Analysis-of-Mental-Health-Social-Media-Text](https://github.com/kktoh1105/Sentiment-Analysis-of-Mental-Health-Social-Media-Text).

## Why This Corrected Version Exists

A forensic analysis of the original repository revealed several critical data leakage issues that invalidated the reported accuracy metrics, particularly the near-perfect 99%+ accuracy reported for BERT models. 

### Key Methodological Fixes

1. **Fixed "Resample BEFORE Split" Leakage:** In the original repository's ML and Deep Learning notebooks, minority classes were heavily upsampled (e.g., from 895 to 16,040 samples) *before* applying `train_test_split`. This meant near-identical copies of rows ended up in both the training and test sets. **Fix:** This repository splits the data *first*, and only resamples the training set.
2. **Unified Evaluation Test Set:** In the original repo, the ML/DL models were tested on a split of a resampled dataset (test size ~22,800), while BERT was tested on a split of the deduplicated original dataset (test size ~5,100). Furthermore, the BERT "Prediction CSV" notebooks used a completely different `train_test_split` logic than the BERT training notebooks, likely causing training data to leak into the test evaluation. **Fix:** All models now draw from a single, unified data splitting pipeline using the exact same held-out test set.
3. **Proper Validation Set for Deep Learning:** Original CNN and BiLSTM notebooks used the test set as the validation set during training for early stopping. **Fix:** A dedicated, separate validation set (10% of total data) is used for all early stopping and hyperparameter tuning. The test set is only touched once at the end.
4. **Near-Duplicate Detection:** Scraped social media data often contains heavy near-duplicates (bots, templates, cross-posts). **Fix:** Added MinHash/Jaccard similarity detection to drop near-duplicates within classes before splitting.

## Project Structure

```text
├── dataset/                  # Place the downloaded kaggle CSVs here
├── results/                  # Generated prediction CSVs are saved here
├── config.py                 # Centralized hyperparameters and settings
├── preprocess.py             # NLP text cleaning and lemmatization utilities
├── data_loader.py            # Unified data splitting and near-duplicate removal
├── train_ml.py               # Script to train and evaluate Logistic Regression, RF, Extra Trees, XGBoost
├── train_deep_learning.py    # Script to train and evaluate BiLSTM, CNN, CNN-BiLSTM
├── train_bert.py             # Script to fine-tune and evaluate BERT
└── evaluate_all.py           # Unified evaluation and comparison script
```

## Getting Started

### 1. Requirements

Ensure you have Python 3.8+ installed. You'll need the following major libraries:
```bash
pip install pandas numpy scikit-learn xgboost tensorflow transformers gensim nltk
```

### 2. Datasets

Download the datasets from Kaggle and place them in the `dataset/` directory. Ensure they are named:
- `dataset1.csv`
- `dataset2.csv`
- `dataset3.csv`

### 3. Running the Models

You can run individual scripts by specifying the dataset:

```bash
# Train ML Models
python train_ml.py --dataset Dataset1

# Train Deep Learning Models
python train_deep_learning.py --dataset Dataset1

# Train BERT Model
python train_bert.py --dataset Dataset1

# Evaluate and compare all generated results
python evaluate_all.py
```

## What to Expect (Realistic Accuracy)

With data leakage removed, you should expect realistic, non-inflated accuracy metrics. The original repository reported 99.6% for BERT and ~95% for ML models. 
With strict data hygiene:
- ML models typically yield **~85-91%** accuracy depending on the dataset.
- BERT and deep learning models typically yield **~90-94%** accuracy.
- Minority classes will have realistic recall/precision scores, rather than the artificial 100% scores caused by duplicated rows leaking into the test set.
