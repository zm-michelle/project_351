from data import get_dataloaders 
from model import Baseline
import torch
import os

DEVICE= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE: ", DEVICE)

epochs = 200 
cur_dir = os.getcwd()

train_dataloader, test_dataloader = get_dataloaders("CIFAR-10")
cifar_100_train_dataloader, cifar_100_test_dataloader = get_dataloaders("CIFAR-100")
svhn_train_dataloader, svhn_test_dataloader = get_dataloaders("SVHN")

model_kwargs =  {
        "input_size": 32 ,
        "in_channels": 3, 
        "hidden_dims": [32, 128, 256, 512, 512, 128],
        "layers_per_block":[1,1,1,1],
        "apply_pooling":[1,1,1,0],
        "apply_bn":[1,1,1,1],
        "num_blocks":4,
        "activation_function": torch.nn.ReLU,
        "dropout_rate": 0.2
} 

run_configs = [
    (
    "CIFAR-10",
        { 
            "logs_dir": os.path.join(cur_dir, 'runs/cifar_10'),
            "model": Baseline,
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
            "epochs": 200,
            "model_kwargs":model_kwargs,
            "optimizer_func":
            lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=1e-5),
            "loss_function": torch.nn.CrossEntropyLoss(),
            "device": DEVICE,
            "use_early_stopper":True,
            "clip_grad_norm":False,
            "weight_init":None,
            "use_scheduler":True,
            "scheduler_kwargs":{"mode":'min', "factor" :0.5, "patience": 5},
            "early_stopping_patience":10,
            "early_stopping_target":98.5,
        }
    ),

    (
    "SVHN",
        { 
            "logs_dir": os.path.join(cur_dir, 'runs/svhn_baseline'),
            "model": Baseline,
            "train_dataloader": svhn_train_dataloader,
            "test_dataloader": svhn_test_dataloader,
            "epochs": 200,
            "model_kwargs":model_kwargs,
            "optimizer_func":
            lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=1e-5),
            "loss_function": torch.nn.CrossEntropyLoss(),
            "device": DEVICE,
            "use_early_stopper":True,
            "clip_grad_norm":False,
            "weight_init":None,
            "use_scheduler":True,
            "scheduler_kwargs":{"mode":'min', "factor" :0.5, "patience": 5},
            "early_stopping_patience":10,
            "early_stopping_target":98.5,
        }
    ),

    (
        "CIFAR-100",
        { 
            "logs_dir": os.path.join(cur_dir, 'runs/cifar_100_baseline'),
            "model": Baseline,
            "train_dataloader":cifar_100_train_dataloader,
            "test_dataloader":cifar_100_test_dataloader,
            "epochs": 200,
            "model_kwargs":model_kwargs,
            "optimizer_func":
            lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=1e-5),
            "loss_function": torch.nn.CrossEntropyLoss(),
            "device": DEVICE,
            "use_early_stopper":True,
            "clip_grad_norm":False,
            "weight_init":None,
            "use_scheduler":True,
            "scheduler_kwargs":{"mode":'min', "factor" :0.5, "patience": 5},
            "early_stopping_patience":10,
            "early_stopping_target":98.5,
        }
    ),
]