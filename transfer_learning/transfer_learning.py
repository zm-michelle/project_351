import torch
import torch.nn as nn
from model import Baseline
from typing import Optional, Dict, Any
import os


def create_transfer_learning_model(
    pretrained_path: str,
    num_classes_source: int = 10,
    num_classes_target: int = 100,
    model_kwargs: Optional[Dict[str, Any]] = None,
    freeze_backbone: bool = True,
    freeze_until_block: Optional[int] = None,
    device: torch.device = torch.device("cpu")):

    # Prepare kwargs for target model
    if model_kwargs is None:
        target_kwargs = {
            "input_size": 32,
            "in_channels": 3,
            "hidden_dims": [32, 128, 256, 512, 512, 128],
            "layers_per_block": [1, 1, 1, 1],
            "apply_pooling": [1, 1, 1, 0],
            "apply_bn": [1, 1, 1, 1],
            "num_blocks": 4,
            "num_classes": num_classes_target,
            "activation_function": torch.nn.ReLU,
            "dropout_rate": 0.2
        }
    else:
        target_kwargs = model_kwargs.copy()
        target_kwargs["num_classes"] = num_classes_target
    
    # Load pre-trained weights
    print(f"Loading pre-trained weights from: {pretrained_path}")
    source_state = torch.load(pretrained_path, map_location=device, weights_only=False)
    
    # Create target model with new number of classes
    target_model = Baseline(**target_kwargs).to(device)
    target_state = target_model.state_dict()
    
    # Transfer weights: transfer convolution blocks and compatible classifier layers
    transferred_keys = []
    skipped_keys = []
    
    for key in source_state.keys():
        # Try to transfer if shapes match
        if key in target_state and source_state[key].shape == target_state[key].shape:
            target_state[key] = source_state[key].clone()
            transferred_keys.append(key)
        else:
            # Skip if shape mismatch
            skipped_keys.append(key)
    
    # Load the updated state dict
    target_model.load_state_dict(target_state)
    
    # Categorize what was transferred for reporting
    conv_keys = [k for k in transferred_keys if 'blocks.' in k]
    fc_keys = [k for k in transferred_keys if 'fc' in k or 'linear' in k.lower()]
    
    print(f"\nTransfer Learning Model Created:")
    print(f"  Source classes: {num_classes_source}")
    print(f"  Target classes: {num_classes_target}")
    print(f"  Transferred weights:")
    print(f"    - Convolutional layers: {len(conv_keys)} parameters")
    if fc_keys:
        print(f"    - Linear layers: {len(fc_keys)} parameters")
    print(f"  Skipped: {len(skipped_keys)} parameters (final classifier + mismatched layers)")
    
    # Freeze layers if requested
    if freeze_backbone:
        if freeze_until_block is not None:
            # Freeze specific blocks
            for block_idx in range(min(freeze_until_block + 1, len(target_model.blocks))):
                for param in target_model.blocks[block_idx].parameters():
                    param.requires_grad = False
            print(f"  Frozen blocks 0 to {freeze_until_block}")
        else:
            # Freeze all convolutional blocks
            for block in target_model.blocks:
                for param in block.parameters():
                    param.requires_grad = False
            print(f"  Frozen all {len(target_model.blocks)} convolutional blocks")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in target_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in target_model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.1f}%)\n")
    
    return target_model

