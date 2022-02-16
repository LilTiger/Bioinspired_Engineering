import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(20, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


linear = LinearRegression()
print(linear)
