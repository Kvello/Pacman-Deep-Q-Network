from collections import namedtuple, deque
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import random
from typing import Tuple, List, Dict, Union




""" Deep Q Network """
class Conv(nn.Module):
    def __init__(self,
                 obs_size:int,
                 num_actions:int,
                 conv_layers:Union[List[int],List[Dict[str,int]]]=[32,64],
                 linear_layers:Union[List[Dict[str,int]],List[int]]=[512],
                 activation:str="ReLU",
                 batch_norm:bool=False):
        super().__init__()
        self.conv_layers = nn.Sequential()
        in_ch = obs_size[0]
        img_size = np.array([obs_size[1], obs_size[2]])
        for i, lay in enumerate(conv_layers):
            if isinstance(lay, int):
                out_ch = lay
                stride = 1
                padding = 0
                kernel_size = 3
            else:
                if "stride" in lay.keys():
                    stride = lay["stride"]
                else:
                    stride = 1
                if "padding" in lay.keys():
                    padding = lay["padding"]
                else:
                    padding = 0
                if "kernel_size" in lay.keys():
                    kernel_size = lay["kernel_size"]
                else:
                    kernel_size = 3
                out_ch = lay["out_channels"]
            self.conv_layers.add_module("conv"+str(i), nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding))
            if batch_norm:
                self.conv_layers.add_module("batch_norm"+str(i), nn.BatchNorm2d(out_ch))
            self.conv_layers.add_module("activation"+str(i), getattr(nn, activation)())
            img_size = (img_size - kernel_size + 2*padding) // stride + 1
            in_ch = out_ch
        self.conv_layers.add_module("flatten", nn.Flatten())
        self.linear_layers = nn.Sequential()
        in_sz = np.prod(img_size)*out_ch 
        for i, out_size in enumerate(linear_layers):
            self.linear_layers.add_module("linear"+str(i), nn.Linear(in_sz, out_size))
            self.linear_layers.add_module("activation"+str(i), getattr(nn, activation)())
            in_sz = out_size
        self.linear_layers.add_module("linear_out", nn.Linear(in_sz, num_actions))

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.linear_layers(out)
        return out

def validate(model:nn.Module,
             valloader:data.DataLoader,
             loss_fn:nn.Module,
             device:str="cpu")->float:
    r"""
    General purpose validation function
    Args:
        model (nn.Module): Model to validate
        valloader (data.DataLoader): Validation dataset
        loss_fn (nn.Module): Loss function to use
        device (str): Device to use for validation. Default: 'cpu'
    """
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for x,y in valloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss += loss_fn(y_pred,y).item()
            pred = y_pred.argmax(dim=1,keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    loss /= len(valloader)

    return loss 
