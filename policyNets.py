import torch
from typing import List, Dict
from torch import nn
import numpy as np
from architectures.feedForward import FeedForward
from architectures.Conv import Conv
from architectures.ResLSTM import LSTM
class PolicyNet(nn.Module):
    def __init__(self,
                 num_object_types,
                 num_actions,
                 obs_size,
                 **kwargs):
        super().__init__()
        args = kwargs["args"]
        args["obs_size"] = (num_object_types,)+obs_size
        args["num_actions"] = num_actions
        if kwargs["arch"].lower() == "feedforward":
            self.net = FeedForward(**args)
        elif kwargs["arch"].lower() == "conv":
            self.net = Conv(**args)
        elif kwargs["arch"].lower() == "lstm":
            self.net = LSTM(**args)
        else:
            raise ValueError("invalid architecture {}".format(kwargs["arch"]))
        self.output  = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.net(x)
        return self.output(x)