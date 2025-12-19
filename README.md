# project_351: Transfer Learning from CIFAR-10 to CIFAR-100 and SVHN

A Pytorch based model and a set of functions that help implement both learning experiments for CIFAR-10, CIFAR-100 and SVHN datasets and transfer learning from CIFAR-10 to CIFAR-100 and SVHN

## Project Structure
```
project_351/
|- main.py # Main training script for baseline(source) models
|- data.py # Gets dataloaders of augmented data for both test and train
|- model.py # Baseline CNN arquitecture
|- config.py # Dictionary configuration for experiments
|- Utils.py # Training Loop and helperfunctions
|- requirements.txt # Requirements
|- transfer_learning/
|---- transfer_configs.py # Dictionary configs for training
|---- transfer_learning # Transfer learning functions
|---- transfer_main #  Main Training script to transfer learning

```
## CNN Arquitecture

Configurable Convolutional Blocks
with optional:
 - Multiple convolutional layers per block
 - Batch Normalization
 - ReLU activation
 - Max Pooling

Classifier: 3 Fully Connected Layeers with Dropout

## Usage

### Dependencies

```bash
pip install -r requirements.txt  
```

Configure your training experiments in the config.py files then:

### For training Baseline Models

```bash
python main.py
```


### For training Transfer Learning Models
WARNING : Make sure the baseline model has already been created and trained

```bash
python transfer_learning/transfer_main.py
```
 

## Configuration Ecample

### Model Configs
Change According to desired network depth
#### Shallow Network

```python
model_kwargs =  {
        "input_size": 32 , # IN this case since both the width and height of the images are the same we keep input size just the size of one.
        "in_channels": 3, # Accounting for each of the RGB channels
        "hidden_dims": [32, 128, 256, 512, 512, 128], # Hidden dimensions for both the convolutional channels and the fully connected layers
        "layers_per_block":[1,1,1,1], # Number of convolutional layers per block
        "apply_pooling":[1,1,1,0], # Whether to apply Max pooling in each block
        "apply_bn":[1,1,1,0], # Whether to apply batch normalization in each block
        "num_blocks":4, # Number of extraction Blocks
        "activation_function": torch.nn.ReLU, 
        "dropout_rate": 0.2
} 
```


#### Deep Network

```python
model_kwargs =  {
        "input_size": 32 , # IN this case since both the width and height of the images are the same we keep input size just the size of one.
        "in_channels": 3, # Accounting for each of the RGB channels
        "hidden_dims": [32, 128, 256, 512, 512,512,512, 128], # Hidden dimensions for both the convolutional channels and the fully connected layers
        "layers_per_block":[1,1,1,1,1,1], # Number of convolutional layers per block
        "apply_pooling":[1,1, 1,1 ,1,0], # Whether to apply Max pooling in each block
        "apply_bn":[1,1,1,1,1,0], # Whether to apply batch normalization in each block
        "num_blocks":6, # Number of extraction Blocks
        "activation_function": torch.nn.ReLU, 
        "dropout_rate": 0.2
} 
```
### Baseline Model Training Configs
Each training experiment, with different learning techniques is stored in a list made up of sets that include both the name of the experiment and the configs.

```python
run_configs = [
    (
    "CIFAR-10", # Name to be used in 'transfer_learning_logs.csv'
        { 
            "logs_dir": os.path.join(cur_dir, 'runs/cifar_10'), # Folder in which the model.pth will be stored alonside the tensorboard files, test accuracies and losses
            "model": Baseline, # A class to be instantiated or an instantiated model
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
            "weight_init":None, # Whether to use He, Xavier, etc initializations
            "use_scheduler":True,
            "scheduler_kwargs":{"mode":'min', "factor" :0.5, "patience": 5}, # change depending on sceduler type
            "early_stopping_patience":10,
            "early_stopping_target":98.5,
        }
    )
]

``` 

## Fine-tuning Instructions
Change the `model` key's value in the dictionary configs and pass an instantiated model to be configured in the following ways:

### Partial Finetune
```python
  "model": create_transfer_learning_model(
                pretrained_path=PRETRAINED_MODEL_PATH,
                num_classes_source=10,
                num_classes_target=100,
                model_kwargs=base_model_kwargs,
                freeze_backbone=True,
                freeze_until_block=1,  # Freeze only first 2 blocks
                device=DEVICE
            ),
```
### Full Finetune
```python
  "model": create_transfer_learning_model(
                pretrained_path=PRETRAINED_MODEL_PATH,
                num_classes_source=10,
                num_classes_target=100,
                model_kwargs=base_model_kwargs,
                freeze_backbone=False,
                device=DEVICE
            ),
```

### Frozen Backbone
```python
"model": create_transfer_learning_model(
                pretrained_path=PRETRAINED_MODEL_PATH,
                num_classes_source=10,
                num_classes_target=100,
                model_kwargs=base_model_kwargs,
                freeze_backbone=True,
                freeze_until_block=None,  # Freeze all conv blocks
                device=DEVICE
            ),
```

## Execution flow

### Baseline Training
1. `main.py` reads experiment configurations from `configs.py`.
2. `data.py` loads and augments CIFAR-10, CIFAR-100, or SVHN datasets.
3. `model.py` instantiates the CNN architecture.
4. `Utils.py` executes the training loop.
5. Model weights are saved to `runs/{chosen name from configs}/model.pth`
6. Training metrics are logged to TensorBoard and CSV files

### Transfer Learning 
1. `transfer_main.py` reads configurations from `transfer_configs.py`
2. `transfer_learning.py` loads pre-trained weights from Phase 1
3. The classifier head is replaced for the new number of classes
4. Selected layers are frozen based on the transfer strategy
5. `Utils.py` fine-tunes the model on the target dataset
6. Results are saved to `runs/{chosen name from configs}/model.pth`
