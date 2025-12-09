import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(
        self,
        input_size=32,
        in_channels=3,
        hidden_dims=[32, 128, 256, 512, 512, 128],
        layers_per_block=[1, 1, 1, 1],
        apply_pooling=[1, 1, 1, 0],
        apply_bn=[1, 1, 1, 1],
        num_blocks=4,
        num_classes=10,
        activation_function=nn.ReLU,
        dropout_rate=0.2,
    ):
        super().__init__()

        assert len(apply_bn) == num_blocks
        assert len(apply_pooling) == num_blocks
        assert len(layers_per_block) == num_blocks
        assert len(hidden_dims)-2 == num_blocks

        self.input_size = input_size
        self.in_channels = in_channels
        self.activation_function = activation_function

        # Build convolutional extraction blocks
        self.blocks = nn.ModuleList()

        current_channels = in_channels
        
        for block_idx in range(num_blocks):
            block_layers = []
            
            for _ in range(layers_per_block[block_idx]):
                out_channels = hidden_dims[block_idx]
                block_layers.append(
                    nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)
                )
                
                if apply_bn[block_idx]:
                    block_layers.append(nn.BatchNorm2d(out_channels))
                
                block_layers.append(activation_function())
                current_channels = out_channels
            
            if apply_pooling[block_idx]:
                block_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            self.blocks.append(nn.Sequential(*block_layers))
        
        # Calculate flat size for linear layer
        self.feature_size = self.get_flat_size()
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.feature_size, hidden_dims[-2])
        self.fc2 = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.fc3 = nn.Linear(hidden_dims[-1], num_classes)
        
    def get_flat_size(self):
        with torch.no_grad():
            x = torch.zeros(1, self.in_channels, self.input_size, self.input_size)
            for block in self.blocks:
                x = block(x)
            return x.view(1, -1).size(1)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.activation_function()(self.fc1(x))
        x = self.dropout(x)
        x = self.activation_function()(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def weight_initializer(self, init_type='xavier'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_type == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
