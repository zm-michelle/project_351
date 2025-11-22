from data import test_dataloader, train_dataloader
from Utils import train_model, train, test
from configs import model_kwargs, run_configs
from matplotlib import pyplot as plt
from datetime import datetime
from model import Baseline
import pandas as pd
import numpy as np
import torch
import csv
import os

EXPERIMENT_LOG_PATH = os.path.join('runs', 'experiment_logs')

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

results = []

result = result = {"title": [], 
              "train_loss":[], 
              "test_loss":[], 
              "test_accuracy":[] }
for run_config in run_configs:
    title, config  = run_config
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d-%H_%M_%S")
    logs_dir = config["logs_dir"]  + '__' + formatted_datetime
    config["logs_dir"] = logs_dir
    model, train_loss, test_loss, test_accuracy = train_model(**config)

      
    torch.save(model.state_dict(), os.path.join(config['logs_dir'],"model.pth" ))

    log_results(title, train_loss, test_loss, test_accuracy ,config)
    result["title"].append(title if title is not None else 'None')
    result["train_loss"].append(train_loss if train_loss is not None else None)
    result["test_loss"].append(test_loss if test_loss is not None else None)
    result["test_accuracy"].append(test_accuracy if test_accuracy is not None else None)
 

print("\n" + "="*50)
print(f"TOTAL RUNS: {len(result['title'])}")
print("="*50)

best_accuracy = max(result["test_accuracy"])
best_idx = result["test_accuracy"].index(best_accuracy)
print(f"\nBEST RUN: {result['title'][best_idx]} | {best_accuracy:.4f}")
print("="*50 + "\n")

fig, axs = plt.subplots(1, 3, figsize=(15, 5)) 
for i in range(len(results["title"])):
    axs[0].plot(result["train_loss"][i], label=result["title"][i])
    axs[1].plot(result["test_loss"][i], label=result["title"][i])
    axs[2].plot(result["test_accuracy"][i], label=result["title"][i])

axs[0].set_title("Train Loss")
axs[1].set_title("Test Loss")
axs[2].set_title("Accuracy")

for ax in axs:
    ax.legend()
    ax.grid() 

plt.tight_layout()
plt.savefig("current_runs.png")

