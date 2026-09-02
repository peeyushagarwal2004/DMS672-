"""
Traditional ML model training: Logistic Regression, Random Forest, Extra Trees, XGBoost.
(Fast execution version: no GridSearch, limited features)
"""

import os
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

import config
from data_loader import prepare_dataset

def train_and_evaluate(model_name, estimator, X_train_tfidf, y_train, 
                       X_test_tfidf, y_test, X_test_raw, label_encoder, dataset_name):
    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"{'='*70}")
    
    estimator.fit(X_train_tfidf, y_train)
    y_pred_enc = estimator.predict(X_test_tfidf)
    
    if model_name == 'XGBoost':
        y_pred = label_encoder.inverse_transform(y_pred_enc)
        y_true = label_encoder.inverse_transform(y_test)
    else:
        y_pred = y_pred_enc
        y_true = y_test
        
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    
    print(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n  Classification Report:\n{report}")
    
    df_pred = pd.DataFrame({'text': X_test_raw.values, 'true_label': y_true, 'predicted_label': y_pred})
    out_path = os.path.join(config.RESULTS_DIR, f"{dataset_name}_{model_name}_predictions.csv")
    df_pred.to_csv(out_path, index=False)
    
    return {'model_name': model_name, 'accuracy': acc, 'report': report}

def main():
    parser = argparse.ArgumentParser(description='Train ML models')
    parser.add_argument('--dataset', default='Dataset1', choices=['Dataset1', 'Dataset2', 'Dataset3'])
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    data = prepare_dataset(args.dataset)
    
    X_train_clean, X_test_clean = data['X_train_clean'], data['X_test_clean']
    y_train, y_test = data['y_train'], data['y_test']

    print("\n  Applying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES)
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)
    X_test_tfidf = vectorizer.transform(X_test_clean)
    print(f"  TF-IDF shapes - Train: {X_train_tfidf.shape} | Test: {X_test_tfidf.shape}")

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    models = [
        {'name': 'LogisticRegression', 'estimator': LogisticRegression(random_state=config.RANDOM_STATE, class_weight='balanced', max_iter=1000), 'use_encoded_labels': False},
        {'name': 'RandomForest', 'estimator': RandomForestClassifier(n_estimators=100, max_depth=30, random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=-1), 'use_encoded_labels': False},
        {'name': 'ExtraTrees', 'estimator': ExtraTreesClassifier(n_estimators=100, max_depth=30, random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=-1), 'use_encoded_labels': False},
        {'name': 'XGBoost', 'estimator': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=config.RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1), 'use_encoded_labels': True}
    ]

    results = []
    for m in models:
        curr_y_train = y_train_enc if m['use_encoded_labels'] else y_train
        curr_y_test = y_test_enc if m['use_encoded_labels'] else y_test
        res = train_and_evaluate(m['name'], m['estimator'], X_train_tfidf, curr_y_train, X_test_tfidf, curr_y_test, data['X_test'], label_encoder, args.dataset)
        results.append(res)
        
    print(f"\n{'='*70}\n  MACHINE LEARNING RESULTS SUMMARY ({args.dataset})\n{'='*70}")
    for r in results:
        print(f"  {r['model_name']:20s} : {r['accuracy']*100:.2f}%")

if __name__ == '__main__':
    main()
