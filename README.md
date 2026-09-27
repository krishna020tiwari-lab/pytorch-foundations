# PyTorch Foundations: From Autograd Mechanics to Production Pipelines

A hands-on, code-first repository tracking my deep learning implementation journey using PyTorch 2.x and CUDA acceleration.

## Progression

1. **`01_cuda_check.py`**: CUDA environment verification and tensor device allocation.
2. **`02_autograd_basics.py`**: Manual computational graphs and backpropagation mechanics (`requires_grad`, `.backward()`).
3. **`03_linear_regression_scratch.py`**: Linear regression built completely from scratch without `torch.nn` modules.
4. **`04_linear_regression_nn.py`**: Refactoring manual linear regression into professional PyTorch abstractions (`nn.Module`, `nn.Linear`, `optim.SGD`).
5. **`05_multi_layer_perceptron.py`**: Non-linear function approximation ($y = x^2$) using Multi-Layer Perceptrons and `ReLU` activation functions.
6. **`06_dataset_dataloader.py`**: Custom `Dataset` and `DataLoader` pipelines for mini-batch stochastic gradient descent.
## Setup & Environment
- PyTorch 2.14.0+cu130
- Python 3.10+
- CUDA Enabled GPU
