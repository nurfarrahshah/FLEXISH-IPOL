FLEXISH: A Learnable Activation Function for Residual Networks Applied to Sea Turtle Individual Identification
================================================================================

This repository contains the official source code for the paper:
"FLEXISH: A Learnable Activation Function for Residual Networks Applied to Sea Turtle Individual Identification" (submitted to Image Processing On Line, IPOL 2025).

This code implements the FLEXISH activation function, integrates it into a ResNet18 architecture, and provides a complete training and evaluation pipeline for benchmarking on an image classification task. The application domain focuses on individual identification of sea turtles, but the framework is generalizable to other classification tasks.

--------------------------------------------------------------------------------
Directory Contents
--------------------------------------------------------------------------------

- `flexish.py`        : Implementation of the FLEXISH activation function.
- `model.py`          : ResNet18 model with customizable activation functions.
- `train.py`          : Training pipeline with logging, metrics, and Flexish parameter tracking.
- `evaluate.py`       : Evaluation script to compute accuracy, F1-score, AUC, and confusion matrices.
- `demo.py`           : Quick demo script for running inference on a single image or folder.
- `requirements.txt`  : Python package dependencies.
- `README.md`         : GitHub-flavored README (web version).
- `LICENSE`           : Licensing information (e.g., MIT or CC-BY-NC-SA).

--------------------------------------------------------------------------------
Installation
--------------------------------------------------------------------------------

1. Clone the repository:
   git clone https://github.com/YourUsername/FLEXISH-IPOL.git
   cd FLEXISH-IPOL

2. Create and activate a Python environment (optional but recommended):
   python3 -m venv env
   source env/bin/activate   (Linux/macOS)
   env\Scripts\activate      (Windows)

3. Install dependencies:
   pip install -r requirements.txt

4. (Optional) Download dataset and pretrained models if available.

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------

To train the model with FLEXISH activation:

   python train.py

To evaluate the model:

   python evaluate.py

To run the demo on an image or folder:

   python demo.py --input path/to/image_or_folder

Adjust parameters inside the scripts or use command-line arguments as needed.


