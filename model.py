import torch
import torch.nn as nn
import torchvision.models as models
from flexish import Flexish, ACTIVATIONS


class ActivationResNet18(nn.Module):
    def __init__(self, num_classes, activation_fn='relu'):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        
        # Get activation function class
        if isinstance(activation_fn, str):
            activation_class = ACTIVATIONS[activation_fn.lower()]
        else:
            activation_class = activation_fn
        
        # Replace all ReLU with custom activation
        self._replace_activations(self.backbone, activation_class)
        
        # Replace final layer
        self.backbone.fc = nn.Linear(512, num_classes)
        
        # Track beta parameters for Flexish
        if activation_class == Flexish:
            self.beta_history = []
    
    def _replace_activations(self, module, activation_fn):
        """Replace all ReLU activations in the model with custom activation"""
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, activation_fn())
            elif len(list(module.children())) > 0:
                self._replace_activations(child, activation_fn)
    
    def track_parameters(self, epoch):
        """Track Flexish parameters during training"""
        if hasattr(self, 'beta_history'):
            beta_values = {}
            for name, module in self.named_modules():
                if isinstance(module, Flexish):
                    beta_values[name] = module.get_beta()
            
            self.beta_history.append({
                'epoch': epoch,
                'beta_values': beta_values
            })
    
    def forward(self, x):
        return self.backbone(x)


def create_model(model_name='resnet18', num_classes=2, activation='flexish'):
    """Factory function to create models"""
    if model_name.lower() == 'resnet18':
        return ActivationResNet18(num_classes, activation)
    else:
        raise ValueError(f"Model {model_name} not supported")
