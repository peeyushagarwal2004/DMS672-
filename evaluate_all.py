"""
Unified evaluation script.
Reads all prediction CSVs from results/ and produces a comparison report.
"""

import os
import sys
import io
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import config


def evaluate_predictions(csv_path: str) -> dict:
    """Evaluate a single prediction CSV file."""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin-1')

    df = df.dropna(subset=['true_label', 'predicted_label'])

    y_true = df['true_label'].astype(str)
    y_pred = df['predicted_label'].astype(str)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    classes = sorted(y_true.unique())

    # Per-class accuracy
    per_class = {}
    for cls in classes:
        mask = y_true == cls
        total = mask.sum()
        correct = (y_pred[mask] == cls).sum()
        per_class[cls] = {'correct': correct, 'total': total, 'accuracy': correct / total if total > 0 else 0}

    return {
        'accuracy': acc,
        'report': report,
        'per_class': per_class,
        'num_samples': len(df),
        'num_classes': len(classes),
        'classes': classes,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate all prediction results')
    parser.add_argument('--results-dir', default=config.RESULTS_DIR, help='Directory with prediction CSVs')
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    # Find all prediction CSVs
    csv_files = sorted([f for f in os.listdir(results_dir) if f.endswith('_predictions.csv')])

    if not csv_files:
        print("No prediction CSV files found in results/")
        return

    summary = []

    for csv_file in csv_files:
        csv_path = os.path.join(results_dir, csv_file)
        name = csv_file.replace('_predictions.csv', '')

        print("=" * 80)
        print(f"  {name}")
        print("=" * 80)

        result = evaluate_predictions(csv_path)

        print(f"  Samples: {result['num_samples']} | Classes: {result['num_classes']}")
        print(f"  Overall Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"\n  Classification Report:")
        for line in result['report'].split('\n'):
            print(f"    {line}")

        print(f"\n  Per-Class Accuracy:")
        for cls, stats in result['per_class'].items():
            print(f"    {cls:30s}: {stats['correct']:5d}/{stats['total']:5d} = {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")

        # Parse dataset and model from filename
        parts = name.split('_', 1)
        dataset = parts[0] if len(parts) > 1 else 'Unknown'
        model = parts[1] if len(parts) > 1 else name

        summary.append({
            'Dataset': dataset,
            'Model': model,
            'Samples': result['num_samples'],
            'Classes': result['num_classes'],
            'Accuracy': round(result['accuracy'] * 100, 2),
        })

    # Summary table
    print("\n\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    # Pivot if possible
    if len(summary_df['Dataset'].unique()) > 1 or len(summary_df['Model'].unique()) > 1:
        try:
            pivot = summary_df.pivot(index='Dataset', columns='Model', values='Accuracy')
            print(f"\n  Accuracy (%) Pivot Table:")
            print(pivot.to_string())
        except Exception:
            pass

    # Best model per dataset
    print("\n  Best Model per Dataset:")
    for ds in summary_df['Dataset'].unique():
        ds_rows = summary_df[summary_df['Dataset'] == ds]
        if len(ds_rows) > 0:
            best = ds_rows.loc[ds_rows['Accuracy'].idxmax()]
            print(f"    {ds}: {best['Model']} with {best['Accuracy']}%")


if __name__ == '__main__':
    main()
