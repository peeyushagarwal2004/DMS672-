"""
Centralized configuration for the Sentiment Analysis project.
All hyperparameters, paths, and model settings are defined here.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASET_PATHS = {
    "Dataset1": os.path.join(DATASET_DIR, "dataset1.csv"),
    "Dataset2": os.path.join(DATASET_DIR, "dataset2.csv"),
    "Dataset3": os.path.join(DATASET_DIR, "dataset3.csv"),
}

# Column names per dataset
# NOTE: dataset2.csv actually has columns [posts, predicted, intensity].
# The label column is "predicted" — "status" does not exist and used to crash the loader.
DATASET_COLUMNS = {
    "Dataset1": {"text": "statement", "label": "status"},
    "Dataset2": {"text": "posts",     "label": "predicted"},
    "Dataset3": {"text": "post",      "label": "status"},
}

# ─── Data Splitting (THE KEY FIX) ───────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2          # 20% held-out test (NEVER touched during training)
VAL_SIZE = 0.125         # 12.5% of remaining 80% = 10% of total for validation

# ─── Text Preprocessing ─────────────────────────────────────────────
MAX_SEQUENCE_LENGTH = 100       # For CNN/BiLSTM
MAX_WORDS = 50000               # Vocabulary size for Keras Tokenizer
BERT_MAX_LEN = 128              # For BERT tokenizer

# ─── ML Model Settings ──────────────────────────────────────────────
TFIDF_MAX_FEATURES = 30000      # Vocabulary cap for TF-IDF (was 5000 — too small)
TFIDF_NGRAM_RANGE = (1, 2)      # Unigrams + bigrams capture negation/short phrases
TFIDF_MIN_DF = 2                # Ignore terms appearing in a single document (noise)

# ─── Deep Learning Settings ─────────────────────────────────────────
EMBEDDING_DIM = 300             # Word2Vec dimension
DL_BATCH_SIZE = 128
DL_EPOCHS = 25
DL_PATIENCE = 4                 # EarlyStopping patience

# ─── BERT Settings ──────────────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"
BERT_BATCH_SIZE = 32
BERT_EPOCHS = 5
BERT_LEARNING_RATE = 2e-5

# ─── Near-Duplicate Detection ───────────────────────────────────────
DUPLICATE_SIMILARITY_THRESHOLD = 0.95  # Jaccard similarity on char-trigram sets
