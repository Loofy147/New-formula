
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * self.d_model**(-1. / 2)
        return x / (rms + self.eps) * self.weight
