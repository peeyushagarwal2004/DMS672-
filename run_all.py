"""
Master script to run all models for a specific dataset and generate the evaluation report.
"""

import os
import sys
import subprocess
import argparse

def run_command(cmd):
    print(f"\n========================================================")
    print(f"Executing: {' '.join(cmd)}")
    print(f"========================================================\n")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='Run full pipeline')
    parser.add_argument('--dataset', default='Dataset1', choices=['Dataset1', 'Dataset2', 'Dataset3'], help='Dataset to process')
    parser.add_argument('--skip-ml', action='store_true', help='Skip ML models')
    parser.add_argument('--skip-dl', action='store_true', help='Skip DL models')
    parser.add_argument('--skip-bert', action='store_true', help='Skip BERT model')
    args = parser.parse_args()

    python_exec = sys.executable

    # Run ML
    if not args.skip_ml:
        run_command([python_exec, "train_ml.py", "--dataset", args.dataset])
        
    # Run DL
    if not args.skip_dl:
        run_command([python_exec, "train_deep_learning.py", "--dataset", args.dataset])
        
    # Run BERT
    if not args.skip_bert:
        run_command([python_exec, "train_bert.py", "--dataset", args.dataset])
        
    # Evaluate All
    run_command([python_exec, "evaluate_all.py"])

if __name__ == '__main__':
    main()
