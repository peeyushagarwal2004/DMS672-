"""
BERT model training for mental health text classification (PyTorch version).

KEY FIXES from original repo:
- Uses the SAME train/val/test split as ML and DL models
- Trains with proper validation set (NOT test set)
- No inconsistent re-splitting for prediction generation
- Predictions generated from the same held-out test set
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

import config
from data_loader import prepare_dataset


class SentimentDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for tokenized text data."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def main():
    parser = argparse.ArgumentParser(description='Train BERT model')
    parser.add_argument('--dataset', default='Dataset1',
                        choices=['Dataset1', 'Dataset2', 'Dataset3'])
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ── Load data using the SAME corrected pipeline as ML/DL ──
    data = prepare_dataset(args.dataset)

    # For BERT, use RAW text (not cleaned) — BERT has its own tokenizer
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']

    # Encode labels to integers
    le = LabelEncoder()
    y_train = le.fit_transform(data['y_train'])
    y_val = le.transform(data['y_val'])
    y_test = le.transform(data['y_test'])
    num_classes = data['num_classes']

    print(f"  Classes: {le.classes_}")

    # ── Tokenize with BERT ──
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    print("  Tokenizing training data...")
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True,
                                max_length=config.BERT_MAX_LEN)
    print("  Tokenizing validation data...")
    val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True,
                              max_length=config.BERT_MAX_LEN)
    print("  Tokenizing test data...")
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True,
                               max_length=config.BERT_MAX_LEN)

    train_dataset = SentimentDataset(train_encodings, y_train)
    val_dataset = SentimentDataset(val_encodings, y_val)
    test_dataset = SentimentDataset(test_encodings, y_test)

    # ── Build and train model ──
    print("\n  Building PyTorch BERT classifier...")
    model = BertForSequenceClassification.from_pretrained(
        config.BERT_MODEL_NAME, num_labels=num_classes
    )

    training_args = TrainingArguments(
        output_dir='./bert_checkpoints',
        num_train_epochs=config.BERT_EPOCHS,
        per_device_train_batch_size=config.BERT_BATCH_SIZE,
        per_device_eval_batch_size=config.BERT_BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        learning_rate=config.BERT_LEARNING_RATE,
        logging_dir='./logs',
        logging_steps=100,
        report_to="none",
    )

    # FIX: Use VALIDATION set for early stopping, NOT test set
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n  Training BERT...")
    trainer.train()

    # ── Evaluate on held-out test set ──
    print("\n  Evaluating on held-out test set...")
    predictions = trainer.predict(test_dataset)
    y_pred_idx = np.argmax(predictions.predictions, axis=1)

    y_pred_labels = le.inverse_transform(y_pred_idx)
    y_true_labels = le.inverse_transform(y_test)

    acc = accuracy_score(y_true_labels, y_pred_labels)
    report = classification_report(y_true_labels, y_pred_labels, zero_division=0)

    print(f"\n  Classification Report:\n{report}")

    # Save predictions (using SAME test set as ML/DL models)
    df_pred = pd.DataFrame({
        'text': X_test.values,
        'true_label': data['y_test'].values,
        'predicted_label': y_pred_labels,
    })
    out_path = os.path.join(config.RESULTS_DIR, f"{args.dataset}_BERT_predictions.csv")
    df_pred.to_csv(out_path, index=False)
    print(f"  Saved predictions to {out_path}")
    print(f"\n  BERT Final Accuracy: {acc*100:.2f}%")


if __name__ == '__main__':
    main()
