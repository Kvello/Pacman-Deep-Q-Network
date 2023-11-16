import torch.nn as nn
from typing import Tuple, Union, List
import numpy as np


class FeedForward(nn.Module):
    """
    A simple feed forward network for Q learning
    Args:
        obs_size (Tuple[int,int]): Size of the observation matrix
        in_chns (int): Number of channels in the input
        num_actions (int): Number of actions in the output
    """
    def __init__(self,
                 obs_size: Tuple[int, int],
                 num_actions: int,
                 layers: List[int],
                 activation: str = "ReLU"):
        super().__init__()
        self.layers = nn.Sequential()
        in_sz = np.prod(obs_size)
        for i, lay in enumerate(layers):
            if lay <= 0:
                raise ValueError("layer size must be positive")
            self.layers.add_module("linear"+str(i), nn.Linear(in_sz, lay))
            self.layers.add_module("activation"+str(i),
                                   getattr(nn, activation)())
            in_sz = lay
        self.layers.add_module(
            "linear"+str(len(layers)), nn.Linear(in_sz, num_actions))

    def forward(self, x):
        return self.layers(x)
