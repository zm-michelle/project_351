from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from typing import List
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torchvision
import random
import torch
import os


def get_dataloaders( dataset_name, batch_size=256):

    # Data Augmentation
    transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
            
        ])  
    
    if dataset_name =="SVHN":
        trainset = datasets.SVHN(
            root='../data',
            split='train',
            download=True,
            transform=transform
        )
        testset = datasets.SVHN(
            root='./data',
            split='test',
            download=True,
            transform=transform
        )

    elif dataset_name == "CIFAR-10":
        trainset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True,
            download=True, 
            transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root='../data', 
            train=False,
            download=True, 
            transform=transform
        )

    elif dataset_name == "CIFAR-100":
        trainset = torchvision.datasets.CIFAR100(
            root='./data', 
            train=True,
            download=True, 
            transform=transform
        )
        testset = torchvision.datasets.CIFAR100(
            root='./data', 
            train=False,
            download=True, 
            transform=transform
        )

    else:
        raise ValueError("Dataset not implemented") 
    
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader