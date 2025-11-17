from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset, Dataset
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
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


class Baseline(torch.nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_dims:List,
                 layers_per_block:List,
                 apply_pooling:List,
                 apply_bn:List,
                 kernel_size:int=3,
                 num_blocks:int=5,
                 num_classes:int=10, 
                 activation_function:torch.nn.Module=torch.nn.ReLU,
                 dropout_rate:float=0.3,
                 ):
        super().__init__()

        assert len(apply_bn) == num_blocks
        assert len(apply_pooling) == num_blocks
        assert len(layers_per_block) == num_blocks
        assert len(hidden_dims)-2 == num_blocks

        self.activation_function = activation_function
        self.blocks = nn.ModuleList()
        in_channels = input_size
    
         
        for i in range(num_blocks):
            block = self.make_block(
                 in_channels=in_channels,
                 out_channels=hidden_dims[i],
                 num_layers=layers_per_block[i],
                 kernel_size=kernel_size ,
                 apply_bn=apply_bn[i],
                 apply_pooling=apply_pooling[i],
             )
            self.blocks.append(block)
            in_channels = hidden_dims[i]
        hidden_dims[i] = self.get_flat_size(input_size).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[i], hidden_dims[i+1]),
            activation_function(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[i+1], hidden_dims[i+2]),
            activation_function(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[i+2], num_classes)     
         )       
    def get_flat_size(self, input_size):
        dummy_input = torch.zeros(1, 3, input_size, input_size)
        for block in self.blocks:
            dummy_input = block(dummy_input)
        return torch.flatten(dummy_input, start_dim=1)
    def make_block(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int,
        apply_bn: bool,
        apply_pooling: bool,
          ) :
        
        layers = []
    
        for i in range(num_layers):
                 
            curr_in = in_channels if i == 0 else out_channels
            
            layers.append(nn.Conv2d(
                curr_in,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ))
            layers.append(self.activation_function())
               
            if apply_pooling:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if apply_bn:
                layers.append(nn.BatchNorm2d(out_channels))   
        return nn.Sequential(*layers)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        flat_x = torch.flatten(x, start_dim=1)

        logits = self.fc(flat_x)
        return logits
    def count_parameters(self):
        total = 0
        for name, param in self.named_parameters():
            total += param.numel()
        return total
    def weight_initializer(self,initialization='he'):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if initialization == 'normal':
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.01) 
                elif initialization == 'xavier':
                    torch.nn.init.xavier_normal_(module.weight)
                elif initialization == 'he':
                    torch.nn.init.kaiming_normal_(module.weight)
                elif initialization == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(module.weight)
                elif initialization == 'he_uniform':
                    torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        return self