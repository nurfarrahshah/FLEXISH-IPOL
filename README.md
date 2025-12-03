# FLEXISH-IPOL
Source code for FLEXISH: A Learnable Activation Function for Residual Networks Applied to Sea Turtle Individual Identification

Overview:
---------
FLEXISH (Flexish: f(x) = x⋅tanh(e^x-1) + β⋅(x⋅σ(x))) is a novel activation function 
with a learnable beta parameter that adapts during training.

Features:
---------
- Learnable parameter (beta) that adapts to data
- Combines benefits of multiple activation functions
- Reduces dead neurons compared to ReLU
- Easy integration with existing PyTorch models

Installation:
-------------
pip install -r requirements.txt

Usage:
------
See demo.py for examples

