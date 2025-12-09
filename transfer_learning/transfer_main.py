import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from transfer_configs import transfer_learning_configs
from Utils import train_model
from main import log_results
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import csv
import os

EXPERIMENT_LOG_PATH = os.path.join('runs', 'transfer_learning_logs.csv')



def run_transfer_learning_experiments():
    results = {
        "title": [],
        "train_loss": [],
        "test_loss": [],
        "test_accuracy": []
    }
    
    print("\n" + "=" * 70)
    print("STARTING TRANSFER LEARNING EXPERIMENTS")
    print("=" * 70 + "\n")
    
    for run_config in transfer_learning_configs:
        title, config = run_config
        
        print("\n" + "-" * 70)
        print(f"RUNNING: {title}")
        print("-" * 70)
        
        now = datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d-%H_%M_%S")
        logs_dir = config["logs_dir"] + '__' + formatted_datetime
        config["logs_dir"] = logs_dir
        
        try:
            # Train & save model
            model, train_loss, test_loss, test_accuracy = train_model(**config)
             
            torch.save(
                model.state_dict(),
                os.path.join(config['logs_dir'], "model.pth")
            )
            
            log_results(title, train_loss, test_loss, test_accuracy, config)
            
            print(f"\n Completed: {title}")
            print(f"  Final Test Accuracy: {test_accuracy[-1]:.2f}%")
            print(f"  Final Test Loss: {test_loss[-1]:.4f}")

            results["title"].append(title)
            results["train_loss"].append(train_loss[-1] if train_loss else None)
            results["test_loss"].append(test_loss[-1] if test_loss else None)
            results["test_accuracy"].append(test_accuracy[-1] if test_accuracy else None)
                  
        except Exception as e:
            print(f"\n Failed: {title}")
            print(f"  Error: {str(e)}")
            results["title"].append(title)
            results["train_loss"].append(None)
            results["test_loss"].append(None)
            results["test_accuracy"].append(None)

    print("\n" + "=" * 70)
    print(f"TRANSFER LEARNING EXPERIMENTS COMPLETE")
    print(f"TOTAL RUNS: {len(results['title'])}")
    print("=" * 70)

    print("\nRESULTS SUMMARY:")
    print("-" * 70)
    for i, title in enumerate(results["title"]):
        acc = results["test_accuracy"][i]
        loss = results["test_loss"][i]
        if acc is not None:
            print(f"{title:50s} | Acc: {acc:6.2f}% | Loss: {loss:.4f}")
        else:
            print(f"{title:50s} | FAILED")
    
    valid_accuracies = [a for a in results["test_accuracy"] if a is not None]
    if valid_accuracies:
        best_accuracy = max(valid_accuracies)
        best_idx = results["test_accuracy"].index(best_accuracy)
        print("\n" + "=" * 70)
        print(f"BEST RESULT: {results['title'][best_idx]}")
        print(f"  Test Accuracy: {best_accuracy:.2f}%")
        print(f"  Test Loss: {results['test_loss'][best_idx]:.4f}")
        print("=" * 70 + "\n")
    
    return results


if __name__ == "__main__":
    results = run_transfer_learning_experiments()