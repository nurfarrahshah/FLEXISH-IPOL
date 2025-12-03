import torch
import torch.nn as nn
import torch.nn.functional as F

class Flexish(nn.Module):
    """
    Flexish: f(x) = x⋅tanh(e^x-1) + β⋅(x⋅σ(x))
    Learnable beta parameter
    """
    def __init__(self, beta_init=0.5):
        super().__init__()
        # Learnable beta parameter
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def forward(self, x):
        # Apply constraint to ensure positive beta
        beta = F.softplus(self.beta)

        # First component: x⋅tanh(e^x-1)
        exp_component = torch.exp(x) - 1
        tanh_component = torch.tanh(exp_component)
        first_term = x * tanh_component

        # Second component: β⋅(x⋅σ(x)) [Swish]
        swish_component = x * torch.sigmoid(x)
        second_term = beta * swish_component

        # Combined activation
        return first_term + second_term

    def get_beta(self):
        """Get current beta value"""
        return F.softplus(self.beta).item()


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Logish(nn.Module):
    """Logish activation: x * log(1 + sigmoid(x))"""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * torch.log(1 + torch.sigmoid(x))


# Dictionary of all activation functions
ACTIVATIONS = {
    'relu': nn.ReLU,
    'swish': Swish,
    'mish': Mish,
    'logish': Logish,
    'flexish': Flexish
}
