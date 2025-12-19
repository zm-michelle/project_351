from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset, Dataset
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import torch.optim as optim
from typing import List
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torchvision
import random
import torch
import os

class EarlyStopper:
    def __init__(self, patience=5, target_val=None):
        
        self.patience = patience
        self.counter =0
        self.min_validation_loss = float('inf')
        self.target =target_val
        
    def early_stop(self, validation_loss, accuracy):
             
        if self.target is not None and accuracy >= self.target:
            return True
        
        if validation_loss <= self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def train(model, 
          dataloader, 
          optimizer,
          loss_function, 
          device, 
          clip_grad_norm=False):
    
    model.train()
    total_loss = 0.0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        loss.backward()
        
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def test(model, 
         dataloader, 
         loss_function, 
         device ):
    correct = 0
    total_loss = 0
    total = 0
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
    
            y_hat = model(x)
            loss = loss_function(y_hat, y)
            total_loss += loss.item()
            
            pred = torch.argmax(y_hat, dim=1)
            predictions.append(pred)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    predictions = torch.cat(predictions).cpu().numpy()
    return avg_loss, accuracy, predictions

def train_model(
    logs_dir,
    model,
    train_dataloader,
    test_dataloader,
    epochs,
    model_kwargs=None,
    optimizer_func=None,
    loss_function= torch.nn.CrossEntropyLoss,
    device= torch.device("cpu"),
    use_early_stopper= False,
    clip_grad_norm=False,
    weight_init=False,
    use_scheduler=False,
    scheduler_kwargs={ "T_max":100, "eta_min":0},
    early_stopping_patience=10,
    early_stopping_target=98.5,
):
    
    os.makedirs(f"{logs_dir}/logs", exist_ok=True)
    os.makedirs(f"{logs_dir}/results", exist_ok=True)
    writer = SummaryWriter(f"{logs_dir}/logs")

    if isinstance(model, type):  # class
        model_kwargs = model_kwargs or {}
        model = model(**model_kwargs).to(device)
    
        if weight_init is not None:
            model.weight_initializer(weight_init)
            
    else:  # instance
        model = model.to(device)

    print(model)
    optimizer = optimizer_func(model.parameters() )
    
    if use_early_stopper:
        early_stopper = EarlyStopper(
            patience=early_stopping_patience, 
            target_val=early_stopping_target )
    
    if use_scheduler:
        if 'mode' in scheduler_kwargs:
             # ReduceLROnPlateau (uses 'mode')
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
            scheduler_type = 'plateau'
        else:
             # CosineAnnealingLR (uses 'T_max')
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
     
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'predictions': [],
    }
 
    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss = train(model, train_dataloader, optimizer, 
                           loss_function, device, clip_grad_norm)
        history['train_loss'].append(train_loss)
        writer.add_scalar(f'Loss/train', train_loss, epoch)
        
        if test_dataloader is not None:
            test_loss, test_accuracy, predictions = test(model, test_dataloader, loss_function, device)
            history['test_loss'].append(test_loss)
            history['test_accuracy'].append(test_accuracy)
            history['predictions'].append(predictions)
            
            writer.add_scalar(f'Loss/test', test_loss, epoch)
            writer.add_scalar(f'Accuracy/test', test_accuracy, epoch)

            if use_scheduler:
                scheduler.step(test_loss)
                
            if use_early_stopper:
                if early_stopper.early_stop(test_loss,accuracy=test_accuracy):
                    print("Early Stopper at epoch {} with accuracy {}% ".format(epoch, test_accuracy))
                    break
            
    writer.close()
    for key, values in history.items():
        if values:        
            np.save(f"{logs_dir}/results/{key}.npy", values)
    if test_dataloader is not None:
        return model, history['train_loss'], history['test_loss'], history['test_accuracy']
    else:
        return model, history['train_loss']