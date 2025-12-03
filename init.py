"""
FLEXISH-IPOL: Flexible Activation Function Package
"""

from .flexish import Flexish, Swish, Mish, Logish, ACTIVATIONS
from .model import ActivationResNet18, create_model
from .train import train_model
from .evaluate import evaluate_model

__version__ = "1.0.0"
__author__ = "nurfarrahshah"
__email__ = "farahfarahim@gmail.com"

__all__ = [
    'Flexish', 'Swish', 'Mish', 'Logish', 'ACTIVATIONS',
    'ActivationResNet18', 'create_model',
    'train_model', 'evaluate_model'
]
