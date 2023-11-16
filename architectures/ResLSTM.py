import torch
import torch.nn as nn
from typing import List, Tuple, Union, Dict
import numpy as np

class ResidualBlock(nn.Module):
    r"""
    Basic residual block for ResNet, as per the original paper https://arxiv.org/pdf/1512.03385.pdf
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride of the first convolutional layer, subsequent layers have stride 1. Default: 1
        kernel_size (int): Size of the convolutional kernels. Default: 3
        padding (int): Amount of padding to use for the convolutional layers. Default: 1
        bias (bool): Whether or not to use bias in the convolutional layers. Default: False
        batchnorm (bool): Whether or not to use batch normalization. Default: True
        dropout_prob (float): Dropout probability, if set to None, no dropout is used. Default: None
        activation (nn.Module): Activation function to use. Default: nn.ReLU()
        depth (int): Number of convolutional layers to use. Default: 2
        padding_mode (str): Padding mode to use for the convolutional layers. Default: 'zeros'
    """

    def __init__(self, in_channels,
                 out_channels, stride=1,
                 kernel_size=3, padding=1,
                 bias=False,
                 batchnorm=True,
                 dropout_prob=None,
                 activation=nn.ReLU(),
                 depth=2,
                 padding_mode='zeros'):
        assert (depth >= 1)
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.depth = depth
        self.activation = activation
        self.dropout_prob = dropout_prob

        # Shortcut connection to match dimensions
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride)

        # Create the residual block layers
        self.layers = nn.ModuleList()
        for i in range(depth):
            in_ch = in_channels if i == 0 else out_channels
            self.layers.append(nn.Conv2d(in_ch, out_channels,
                                         kernel_size, stride if i == 0 else 1,
                                         padding, padding_mode=padding_mode,
                                         bias=bias))
            if batchnorm:
                self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(activation)
            if dropout_prob is not None:
                self.layers.append(nn.Dropout2d(p=dropout_prob))

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = x
        for layer in self.layers:
            out = layer(out)

        out = out + shortcut
        return self.activation(out)


class ResidualNetwork(nn.Module):
    r"""
    General ResNet model used for classification. Following the general ideas of https://arxiv.org/pdf/1512.03385.pdf
    Args:
        num_classes (int): Number of classes to predict
        image_size (Tuple[int, int, int]): Size of the input images. Default: (3, 224, 224)
        start_block (Union[Tuple[int, int, int, int], nn.Module]): Either a tuple of the form (out_channels, kernel_size, stride, padding) or a nn.Module object. This block should be a convolutional layer.
        Default: (64,7,2) (ResNet-18)
        block_list (Union[List[Tuple[int, int, int, int]], List[ResidualBlock]]): List of tuples of the form (out_channels, stride, depth) 
        or ResidualBlock objects. Default: [(64,1,2),(64, 2, 2), (128, 1, 2), (128,2,2),(256,1,2),(256, 2, 2),(512,1,2) (512, 2, 2)] (ResNet-18)
        kernel_size (int): Size of the convolutional kernels. Default: 3
        padding (int): Amount of padding to use for the convolutional layers. Default: 1
        bias (bool): Whether or not to use bias in the convolutional layers. Default: False
        batchnorm (bool): Whether or not to use batch normalization. Default: True
        dropout_prob (float): Dropout probability, if set to None, no dropout is used. Default: None
        activation (nn.Module): Activation function to use. Default: nn.ReLU()
        padding_mode (str): Padding mode to use for the convolutional layers. Default: 'zeros'
    """
    # TODO: Add support for different kernel sizes and paddings?

    def __init__(self,
                 img_size: Tuple[int, int, int],
                 start_block: Union[Tuple[int, int, int,int],nn.Module] = (64, 7, 2, 1),
                 block_list: Union[List[Tuple[int, int, int]], List[ResidualBlock]] = [(64, 1, 2), (64, 2, 2), (128, 1, 2), (128, 2, 2),(256, 1, 2), (256, 2, 2), (512, 1, 2), (512, 2, 2)],
                 bias=False,
                 batchnorm=True, dropout_prob=None,
                 activation=nn.ReLU(),
                 padding_mode='zeros'):
        super().__init__()

        self.in_channels = img_size[0]
        self.bias = bias
        self.batchnorm = batchnorm
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.padding_mode = padding_mode

        self.blocks = nn.ModuleList()
        img_size = np.array(img_size[1:])
        if isinstance(start_block, nn.Module):
            self.blocks.append(start_block)
            img_size = (img_size-start_block.dialation\
                        *(start_block.kernel_size-1)\
                        -1+2*start_block.padding)\
                        //start_block.stride+1
        else:
            ker_size = start_block[1]
            out_channels = start_block[0]
            stride = start_block[2]
            padding = start_block[3]
            self.blocks.append(nn.Conv2d(
                self.in_channels, out_channels=out_channels, kernel_size=ker_size, stride=stride, padding=padding))
            img_size = (img_size-ker_size+2*padding)//stride+1
        self.blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        img_size = (img_size-3+2*1)//2+1
        in_chn = start_block[0]
        for block in block_list:
            if isinstance(block, nn.Module):
                self.add_block(block)
                img_size = (img_size-block.dialation*(block.kernel_size-1) -\
                            1+2*block.padding)//block.stride+1
                in_chn = block.out_channels
            else:
                out_channels, stride, depth = block
                ker_size = 3
                padding = 1
                self.blocks.append(ResidualBlock(in_chn,
                                                 out_channels, stride,
                                                 kernel_size=ker_size,
                                                 padding=padding, bias=bias, batchnorm=batchnorm,
                                                 dropout_prob=dropout_prob, activation=activation,
                                                 depth=depth, padding_mode=padding_mode))
                img_size = img_size//stride
                in_chn = out_channels
        self.blocks.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        img_size = (img_size-3+2*1)//2+1

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out

class LSTM(nn.Module):
    def __init__(self,
                 obs_size:int,
                 num_actions:int,
                 hidden_size:int,
                 num_layers:int,
                 pre_process_layers:List[int],
                 post_process_layers:List[int],
                 bidirectional:bool=False,
                 batch_first:bool=True,
                 dropout:float=0.0,
                 device:str="cpu"
                 ):
        super().__init__()
        self.device = device
        self.dense_pre= nn.Sequential()
        in_sz = np.prod(obs_size)
        for i, lay in enumerate(pre_process_layers):
            self.dense_pre.add_module("linear_pre"+str(i), nn.Linear(in_sz, lay))
            self.dense_pre.add_module("activation_pre"+str(i), nn.ReLU())
            in_sz = lay
        self.lstm = nn.LSTM(input_size=in_sz,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=batch_first,
                            dropout=dropout)
        self.dense_post = nn.Sequential()
        in_sz = hidden_size
        if bidirectional:
            in_sz *= 2
        for i, lay in enumerate(post_process_layers):
            self.dense_post.add_module("linear_post"+str(i), nn.Linear(in_sz, lay))
            self.dense_post.add_module("activation_post"+str(i), nn.ReLU())
            in_sz = lay
        self.dense_post.add_module("linear_out", nn.Linear(in_sz, num_actions))
        self.cell = torch.zeros((self.lstm.num_layers,self.lstm.hidden_size),device=self.device).double()
        self.hidden = torch.zeros((self.lstm.num_layers,self.lstm.hidden_size),device=self.device).double()
    def forward(self, x, seq_len=None):
        out = self.dense_pre(x)
        if not self.training and seq_len is None:
            out,_ = self.lstm(out,(self.hidden,self.cell))
        else:
            if seq_len is not None:
                padded_length = x.shape[1]
                out = nn.utils.rnn.pack_padded_sequence(out,seq_len,batch_first=True,enforce_sorted=False)
            out,_ = self.lstm(out)
            if seq_len is not None:
                out,_= nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=padded_length)
        out = self.dense_post(out)
        return out
    def reset(self):
        self.cell = torch.zeros((self.lstm.num_layers,self.lstm.hidden_size),device=self.device).double()
        self.hidden = torch.zeros((self.lstm.num_layers,self.lstm.hidden_size),device=self.device).double()

    
class ResLSTM(nn.Module):
    def __init__(self,
                obs_size:Tuple[int,int],
                num_actions:int,
                resNetConf: Dict[str,Union[int,bool,float]],
                lstmConf:Dict[str,Union[int,bool,float]],
                device:str="cpu"):

        super().__init__()
        self.device = device
        self.batch_first = lstmConf["batch_first"]
        self.resnet = ResidualNetwork(img_size=(obs_size),
                                      **resNetConf).to(device)
        self.resnet.eval()
        resNetOutSize = self.resnet(torch.zeros((1,obs_size[0],obs_size[1],obs_size[2]))).shape
        self.lstm = LSTM(obs_size=resNetOutSize[1:],
                                  actions = num_actions,
                                **lstmConf)
    def forward(self,x):
        out = self.resnet(x)
        if len(out.shape) == 3:
            out = out.unsqueeze(0)
        out = out.view(out.size(0),-1)
        if len(out.shape) == 2:
            seq_dim = 1 if self.batch_first else 0
            out = out.unsqueeze(seq_dim) 
        out = self.lstm(out)
        return out