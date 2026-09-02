# Sentiment Analysis of Mental Health Social Media Text

An end-to-end NLP pipeline for multi-class classification of social media posts into 7 mental health categories. Benchmarks 8 ML and Deep Learning models including fine-tuned BERT, achieving **80.81% accuracy** on a strictly held-out test set.

## Results

| Model | Accuracy |
|:---|:---:|
| **BERT** | **80.81%** |
| Logistic Regression | 75.00% |
| BiLSTM | 74.39% |
| XGBoost | 73.32% |
| CNN-BiLSTM | 72.46% |
| Random Forest | 71.20% |
| CNN | 69.52% |
| Extra Trees | 68.96% |

## Dataset

- **51,000+** social media posts across **7 mental health categories**: Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, Suicidal
- Sourced from Kaggle (see `dataset/` folder)
- Severe class imbalance handled via post-split resampling (minority class: 895 samples vs majority: 16,039)

## Project Structure

```
├── dataset/                     # Raw CSV datasets
├── results/                     # Model prediction CSVs
├── config.py                    # Centralized hyperparameters and settings
├── preprocess.py                # Text cleaning and stopword removal
├── data_loader.py               # Data pipeline: loading, splitting, resampling
├── train_ml.py                  # Logistic Regression, Random Forest, Extra Trees, XGBoost
├── train_deep_learning.py       # BiLSTM, CNN, CNN-BiLSTM
├── train_bert.py                # BERT fine-tuning (PyTorch + HuggingFace)
├── evaluate_all.py              # Unified evaluation and comparison
├── run_all.py                   # End-to-end runner
└── requirements.txt
```

## Key Technical Highlights

- **Leakage-free data pipeline**: Stratified train/validation/test split applied *before* any class resampling, with near-duplicate detection using MinHash/Jaccard similarity
- **Zero data overlap** verified programmatically across train, validation, and test sets
- **Proper validation strategy**: Dedicated validation set (10%) for early stopping and hyperparameter tuning — test set only touched once for final evaluation
- **Multiple embedding strategies**: TF-IDF for classical ML, Word2Vec (300d) for deep learning, BERT tokenization for transformer models

## Getting Started

### Requirements
```bash
pip install -r requirements.txt
```

### Run Individual Models
```bash
python train_ml.py --dataset Dataset1
python train_deep_learning.py --dataset Dataset1
python train_bert.py --dataset Dataset1
```

### Run Full Pipeline
```bash
python run_all.py --dataset Dataset1
```

### Evaluate All Results
```bash
python evaluate_all.py
```

## Tech Stack

Python, PyTorch, TensorFlow, HuggingFace Transformers, Scikit-learn, XGBoost, Gensim, NLTK
