import torch
from torch import nn
def num_parameters(m):
    return sum([p.numel() for p in m.parameters()])
dk, m, n = 3, 16, 32
print(f"Expected number of parameters: {m * dk * dk * n}")
conv1 = nn.Conv2d(in_channels=m, out_channels=n, kernel_size=dk, bias=False)
print(f"Actual number of parameters: {num_parameters(conv1)}")