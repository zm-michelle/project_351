from Utils import train_model, train, test
from configs import run_configs
from matplotlib import pyplot as plt
from datetime import datetime
from model import Baseline
import pandas as pd
import numpy as np
import torch
import csv
import os

EXPERIMENT_LOG_PATH = os.path.join('runs', 'experiment_logs.csv')

def log_results(title, train_loss, test_loss, test_accuracy ,config):

    row = {
        "description": title,
        "test_accuracy": test_accuracy,
        "train_loss": train_loss,
        "test_loss": test_loss,
    }
    for key, val in config.items():
        row[key] = val
         
    
    if not os.path.exists(EXPERIMENT_LOG_PATH): 
        df = pd.DataFrame([row])
    else:
        df = pd.read_csv(EXPERIMENT_LOG_PATH)
        columns = df.columns

        for key in row.keys(): 
            if key not in columns: 
                df[key] = np.nan  
        
        df.loc[len(df)] = row
    df.to_csv(EXPERIMENT_LOG_PATH, index=False)


def run_experiments():
    results = {
        "title": [], 
        "train_loss":[], 
        "test_loss":[], 
        "test_accuracy":[] 
    }

    print("\n" + "=" * 70)
    print("STARTING EXPERIMENTS")
    print("=" * 70 + "\n")

    for run_config in run_configs:
        title, config  = run_config

        print("\n" + "-" * 70)
        print(f"RUNNING: {title}")
        print("-" * 70)
     
    
        now = datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d-%H_%M_%S")
        logs_dir = config["logs_dir"]  + '__' + formatted_datetime
        config["logs_dir"] = logs_dir

        try:
            # Train & save model
            model, train_loss, test_loss, test_accuracy = train_model(**config)

            torch.save(
                model.state_dict(), 
                os.path.join(config['logs_dir'],"model.pth" ))

            log_results(title, train_loss, test_loss, test_accuracy ,config)

            print(f"\n Completed: {title}")
            print(f"  Final Test Accuracy: {test_accuracy[-1]:.2f}%")
            print(f"  Final Test Loss: {test_loss[-1]:.4f}")

            results["title"].append(title if title is not None else 'None')
            results["train_loss"].append(train_loss[-1] if train_loss is not None else None)
            results["test_loss"].append(test_loss[-1] if test_loss is not None else None)
            results["test_accuracy"].append(test_accuracy[-1] if test_accuracy is not None else None)
        except Exception as e:
            print(f"\n Failed: {title}")
            print(f"  Error: {str(e)}")
            results["title"].append(title)
            results["train_loss"].append(None)
            results["test_loss"].append(None)
            results["test_accuracy"].append(None)

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print(f"TOTAL RUNS: {len(results['title'])}")
    print("="*70)

    print("\nRESULTS SUMMARY:")
    print("-" * 70)
    for i, title in enumerate(results['title']):
        acc = results["test_accuracy"][i]
        loss = results["test_loss"][i]

        if acc is not None:
            print(f"{title:50s} | Acc: {acc:6.2f}% | Loss: {loss:.4f}")
        else:
            print(f"{title:50s} | FAILED")

    valid_accuracies = [a for a in results["test_accuracy"] if a is not None]
    if valid_accuracies:
        best_accuracy = max(results["test_accuracy"])
        best_idx = results["test_accuracy"].index(best_accuracy)
        print(f"\nBEST RUN: {results['title'][best_idx]} | {best_accuracy:.4f}")
        print("="*50 + "\n")

if __name__ == "__main__":
    results = run_experiments()

