"""
Deep Learning model training: BiLSTM, CNN, CNN-BiLSTM.

KEY FIXES from original repo:
- Uses VALIDATION set for early stopping (NOT the test set)
- Uses data from the corrected data_loader (split-first pipeline)
- Consistent test set across all models
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

import config
from data_loader import prepare_dataset


def build_bilstm(num_classes, embedding_matrix, max_words, max_seq_len):
    """Build a Bidirectional LSTM model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

    model = Sequential([
        Embedding(min(max_words, embedding_matrix.shape[0]),
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix],
                  input_length=max_seq_len,
                  trainable=True),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn(num_classes, embedding_matrix, max_words, max_seq_len):
    """Build a CNN model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Embedding, SpatialDropout1D, Conv1D,
                                         BatchNormalization, GlobalMaxPooling1D,
                                         Dense, Dropout)

    model = Sequential([
        Embedding(min(max_words, embedding_matrix.shape[0]),
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix],
                  input_length=max_seq_len,
                  trainable=True),
        SpatialDropout1D(0.2),
        Conv1D(128, 5, activation='relu'),
        BatchNormalization(),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_bilstm(num_classes, embedding_matrix, max_words, max_seq_len):
    """Build a hybrid CNN-BiLSTM model."""
    from tensorflow.keras.layers import (Input, Embedding, SpatialDropout1D, Conv1D,
                                         BatchNormalization, Dropout,
                                         Bidirectional, LSTM, GlobalMaxPooling1D,
                                         Dense, concatenate)
    from tensorflow.keras.models import Model

    inp = Input(shape=(max_seq_len,))
    x = Embedding(min(max_words, embedding_matrix.shape[0]),
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix],
                  trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)

    # CNN branch
    cnn = Conv1D(128, 5, activation='relu')(x)
    cnn = BatchNormalization()(cnn)
    cnn = GlobalMaxPooling1D()(cnn)

    # BiLSTM branch
    lstm = Bidirectional(LSTM(128, return_sequences=False))(x)

    # Merge
    merged = concatenate([cnn, lstm])
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    out = Dense(num_classes, activation='softmax')(merged)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_embedding_matrix(tokenizer_word_index, max_words, embedding_dim):
    """Load Word2Vec and build embedding matrix."""
    from gensim.downloader import load as gensim_load

    print("  Loading Word2Vec embeddings (this may take a while on first run)...")
    w2v = gensim_load("word2vec-google-news-300")

    num_words = min(max_words, len(tokenizer_word_index) + 1)
    matrix = np.zeros((num_words, embedding_dim))
    found = 0

    for word, idx in tokenizer_word_index.items():
        if idx >= num_words:
            continue
        if word in w2v:
            matrix[idx] = w2v[word]
            found += 1
        else:
            matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    print(f"  Found embeddings for {found}/{num_words} words")
    return matrix


def train_and_evaluate(model, model_name, X_train_pad, y_train_cat, X_val_pad,
                       y_val_cat, X_test_pad, y_test_cat, label_encoder,
                       X_test_raw, y_test_raw, dataset_name):
    """Train a model and evaluate on test set."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"{'='*70}")

    # FIX: Use VALIDATION set for callbacks, NOT the test set
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.DL_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    ]

    history = model.fit(
        X_train_pad, y_train_cat,
        epochs=config.DL_EPOCHS,
        batch_size=config.DL_BATCH_SIZE,
        validation_data=(X_val_pad, y_val_cat),   # <-- VALIDATION set, not test!
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on HELD-OUT test set
    test_loss, test_acc = model.evaluate(X_test_pad, y_test_cat, verbose=0)
    print(f"\n  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Predictions
    y_pred_probs = model.predict(X_test_pad, verbose=0)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    y_true_idx = np.argmax(y_test_cat, axis=1)
    y_pred_labels = label_encoder.inverse_transform(y_pred_idx)
    y_true_labels = label_encoder.inverse_transform(y_true_idx)

    acc = accuracy_score(y_true_labels, y_pred_labels)
    report = classification_report(y_true_labels, y_pred_labels, zero_division=0)
    print(f"\n  Classification Report:\n{report}")

    # Save predictions
    df_pred = pd.DataFrame({
        'text': X_test_raw.values,
        'true_label': y_test_raw.values,
        'predicted_label': y_pred_labels
    })
    out_path = os.path.join(config.RESULTS_DIR, f"{dataset_name}_{model_name}_predictions.csv")
    df_pred.to_csv(out_path, index=False)
    print(f"  Saved predictions to {out_path}")

    return {'model_name': model_name, 'accuracy': acc, 'report': report}


def main():
    parser = argparse.ArgumentParser(description='Train deep learning models')
    parser.add_argument('--dataset', default='Dataset1', choices=['Dataset1', 'Dataset2', 'Dataset3'])
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Prepare data
    data = prepare_dataset(args.dataset)

    # Tokenize and pad
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical

    tokenizer = Tokenizer(num_words=config.MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['X_train_clean'])  # Fit on TRAINING only

    X_train_seq = tokenizer.texts_to_sequences(data['X_train_clean'])
    X_val_seq = tokenizer.texts_to_sequences(data['X_val_clean'])
    X_test_seq = tokenizer.texts_to_sequences(data['X_test_clean'])

    X_train_pad = pad_sequences(X_train_seq, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(data['y_train'])
    y_val_enc = label_encoder.transform(data['y_val'])
    y_test_enc = label_encoder.transform(data['y_test'])

    y_train_cat = to_categorical(y_train_enc, num_classes=data['num_classes'])
    y_val_cat = to_categorical(y_val_enc, num_classes=data['num_classes'])
    y_test_cat = to_categorical(y_test_enc, num_classes=data['num_classes'])

    print(f"  Padded shapes - Train: {X_train_pad.shape} | Val: {X_val_pad.shape} | Test: {X_test_pad.shape}")

    # Load embeddings
    embedding_matrix = load_embedding_matrix(
        tokenizer.word_index, config.MAX_WORDS, config.EMBEDDING_DIM
    )

    # Train each model
    results = []
    model_builders = [
        ('BiLSTM', build_bilstm),
        ('CNN', build_cnn),
        ('CNN-BiLSTM', build_cnn_bilstm),
    ]

    for model_name, builder in model_builders:
        model = builder(data['num_classes'], embedding_matrix,
                        config.MAX_WORDS, config.MAX_SEQUENCE_LENGTH)
        result = train_and_evaluate(
            model, model_name, X_train_pad, y_train_cat,
            X_val_pad, y_val_cat, X_test_pad, y_test_cat,
            label_encoder, data['X_test'], data['y_test'], args.dataset
        )
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(f"  DEEP LEARNING RESULTS SUMMARY ({args.dataset})")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['model_name']:20s} : {r['accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()
