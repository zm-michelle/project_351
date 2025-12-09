import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data import get_dataloaders
from transfer_learning import create_transfer_learning_model
import torch
import os

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE: ", DEVICE)

epochs = 200
cur_dir = os.getcwd()

train_dataloader, test_dataloader = get_dataloaders("CIFAR-10")
cifar_100_train_dataloader, cifar_100_test_dataloader = get_dataloaders("CIFAR-100")
svhn_train_dataloader, svhn_test_dataloader = get_dataloaders("SVHN")

base_model_kwargs = {
    "input_size": 32,
    "in_channels": 3,
    "hidden_dims": [32, 128, 256, 512, 512, 128],
    "layers_per_block": [1, 1, 1, 1],
    "apply_pooling": [1, 1, 1, 0],
    "apply_bn": [1, 1, 1, 1],
    "num_blocks": 4,
    "activation_function": torch.nn.ReLU,
    "dropout_rate": 0.2
}

PRETRAINED_MODEL_PATH = "runs/cifar_10/model.pth"  

transfer_learning_configs = [
    (
        "CIFAR-10 -> CIFAR-100 (Frozen Backbone)",
        {
            "logs_dir": os.path.join(cur_dir, 'runs/transfer_cifar10_to_100_frozen'),
            "model": create_transfer_learning_model(
                pretrained_path=PRETRAINED_MODEL_PATH,
                num_classes_source=10,
                num_classes_target=100,
                model_kwargs=base_model_kwargs,
                freeze_backbone=True,
                freeze_until_block=None,  # Freeze all conv blocks
                device=DEVICE
            ),
            "train_dataloader": cifar_100_train_dataloader,
            "test_dataloader": cifar_100_test_dataloader,
            "epochs": 100,
            "model_kwargs": None,   
            "optimizer_func": lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=1e-5),
            "loss_function": torch.nn.CrossEntropyLoss(),
            "device": DEVICE,
            "use_early_stopper": True,
            "clip_grad_norm": False,
            "weight_init": None,
            "use_scheduler": False,   
            "scheduler_kwargs": {"T_max": 100, "eta_min": 1e-6},
            "early_stopping_patience": 10,
            "early_stopping_target": 95.0,
        }
    ),
 
    (
        "CIFAR-10 -> CIFAR-100 (Partial)",
        {
            "logs_dir": os.path.join(cur_dir, 'runs/transfer_cifar10_to_100_partial'),
            "model": create_transfer_learning_model(
                pretrained_path=PRETRAINED_MODEL_PATH,
                num_classes_source=10,
                num_classes_target=100,
                model_kwargs=base_model_kwargs,
                freeze_backbone=True,
                freeze_until_block=1,  # Freeze only first 2 blocks
                device=DEVICE
            ),
            "train_dataloader": cifar_100_train_dataloader,
            "test_dataloader": cifar_100_test_dataloader,
            "epochs": 150,
            "model_kwargs": None,
            "optimizer_func": lambda x: torch.optim.Adam(
                [p for p in x if p.requires_grad], 
                lr=1e-3, 
                weight_decay=1e-5
            ),
            "loss_function": torch.nn.CrossEntropyLoss(),
            "device": DEVICE,
            "use_early_stopper": True,
            "clip_grad_norm": False,
            "weight_init": None,
            "use_scheduler": False,   
            "scheduler_kwargs": {"T_max": 150, "eta_min": 1e-6},
            "early_stopping_patience": 15,
            "early_stopping_target": 95.0,
        }
    ),  

   (
        "CIFAR-10 -> CIFAR-100 (Full Finetune)",
        {
            "logs_dir": os.path.join(cur_dir, 'runs/transfer_cifar10_to_100_finetune'),
            "model": create_transfer_learning_model(
                pretrained_path=PRETRAINED_MODEL_PATH,
                num_classes_source=10,
                num_classes_target=100,
                model_kwargs=base_model_kwargs,
                freeze_backbone=False,  # Don't freeze anything
                device=DEVICE
            ),
            "train_dataloader": cifar_100_train_dataloader,
            "test_dataloader": cifar_100_test_dataloader,
            "epochs": 200,
            "model_kwargs": None,
            "optimizer_func": lambda x: torch.optim.Adam(x, lr=5e-4, weight_decay=1e-5),
            "loss_function": torch.nn.CrossEntropyLoss(),
            "device": DEVICE,
            "use_early_stopper": True,
            "clip_grad_norm": False,
            "weight_init": None,
            "use_scheduler": False,  # Disabled - scheduler.step() incompatible with CosineAnnealingLR
            "scheduler_kwargs": {"T_max": 200, "eta_min": 1e-6},
            "early_stopping_patience": 15,
            "early_stopping_target": 95.0,
        }
    ),

    (
        "CIFAR-10 -> SVHN (Frozen Backbone)",
        { 
            "logs_dir": os.path.join(cur_dir, 'runs/transfer_cifar10_to_svhn_frozen'),
            "model": create_transfer_learning_model(
                pretrained_path=PRETRAINED_MODEL_PATH,
                num_classes_source=10,
                num_classes_target=10,   
                model_kwargs=base_model_kwargs,
                freeze_backbone=True,
                freeze_until_block=None, # Freeze all blocks
                device=DEVICE
            ),
            "train_dataloader": svhn_train_dataloader,
            "test_dataloader": svhn_test_dataloader,
            "epochs": 100,
            "model_kwargs": None,
            "optimizer_func": lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=1e-5),
            "loss_function": torch.nn.CrossEntropyLoss(),
            "device": DEVICE,
            "use_early_stopper": True,
            "clip_grad_norm": False,
            "weight_init": None,
            "use_scheduler": False,   
            "scheduler_kwargs": {"T_max": 100, "eta_min": 1e-6},
            "early_stopping_patience": 10,
            "early_stopping_target": 98.0,
        }
    ),
]
