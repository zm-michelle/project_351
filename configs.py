from data import test_dataloader, train_dataloader
from Utils import train_model, train, test
from model import Baseline
import torch
import os

DEVICE= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE: ", DEVICE)
epochs = 200 
cur_dir = os.getcwd()

model_kwargs = [
    {
        "input_size": 32 ,
        "in_channels": 3, 
        "hidden_dims": [32, 64, 128, 256, 512, 512, 128],
        "layers_per_block":[1,1,1,1,1],
        "apply_pooling":[1,1,1,1,0],
        "apply_bn":[1,1,1,1,1],
        "num_blocks":5,
        "num_classes": 10, 
        "activation_function": torch.nn.ReLU,
        "dropout_rate": 0.2
    },
    {
        "input_size": 32 ,
        "in_channels": 3, 
        "hidden_dims": [32, 64, 128, 256, 512, 512, 128],
        "layers_per_block":[1,1,1,1,1],
        "apply_pooling":[1,1,1,1,0],
        "apply_bn":[1,1,1,1,1],
        "num_blocks":5,
        "num_classes": 10, 
        "activation_function": torch.nn.ReLU,
        "dropout_rate": 0.2
    },
]

run_configs = [
    (
    "he_init - run1",
    { "logs_dir": os.path.join(cur_dir, 'runs/run1'),
    "model": Baseline,
    "train_dataloader":train_dataloader,
    "test_dataloader":test_dataloader,
    "epochs": 200,
    "model_kwargs":model_kwargs[0],
    "optimizer_func":
    lambda x: torch.optim.Adam(x, lr=1e-4, weight_decay=1e-5),
    "loss_function": torch.nn.CrossEntropyLoss(),
    "device": DEVICE,
    "use_early_stopper":True,
    "clip_grad_norm":False,
    "weight_init":"he",
    "use_scheduler":True,
    "scheduler_kwargs":{"mode":'min', "factor" :0.5, "patience": 5},
    "early_stopping_patience":10,
    "early_stopping_target":98.5,}),

    (
    "no init - run2",
    { "logs_dir": os.path.join(cur_dir, 'runs/run2'),
    "model": Baseline,
    "train_dataloader":train_dataloader,
    "test_dataloader":test_dataloader,
    "epochs": 200,
    "model_kwargs":model_kwargs[0],
    "optimizer_func":
    lambda x: torch.optim.Adam(x, lr=1e-4, weight_decay=1e-5),
    "loss_function": torch.nn.CrossEntropyLoss(),
    "device": DEVICE,
    "use_early_stopper":True,
    "clip_grad_norm":False,
    "weight_init":None,
    "use_scheduler":True,
    "scheduler_kwargs":{"mode":'min', "factor" :0.5, "patience": 5},
    "early_stopping_patience":10,
    "early_stopping_target":98.5,}),
    (
    "shallow network - run3",
    { "logs_dir": os.path.join(cur_dir, 'runs/run3'),
    "model": Baseline,
    "train_dataloader":train_dataloader,
    "test_dataloader":test_dataloader,
    "epochs": 200,
    "model_kwargs":model_kwargs[0],
    "optimizer_func":
    lambda x: torch.optim.Adam(x, lr=1e-4, weight_decay=1e-5),
    "loss_function": torch.nn.CrossEntropyLoss,
    "device": DEVICE,
    "use_early_stopper":True,
    "clip_grad_norm":False,
    "weight_init":None,
    "use_scheduler":True,
    "scheduler_kwargs":{"mode":'min', "factor" :0.5, "patience": 5},
    "early_stopping_patience":10,
    "early_stopping_target":98.5,}),
]