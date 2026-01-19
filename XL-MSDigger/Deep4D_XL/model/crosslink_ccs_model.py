import copy
from typing import Optional, Any
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class Transformer(Module):

    def __init__(self, feature_len=23, max_len=50, d_model: int = 500, nhead: int = 5, num_encoder_layers: int = 5,
                 dim_feedforward: int = 1200, dropout: float = 0, activation: str = "relu"):
        super(Transformer, self).__init__()
        encoder_layer1 = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_layer2 = TransformerEncoderLayer(d_model, nhead, dim_feedforward, 0, activation)
                                                                                                          
        encoder_norm = LayerNorm(d_model)
        self.linear1 = Linear(feature_len, d_model)                                                               
        self.charge_encoder1 = Linear(1, max_len)
        self.charge_encoder2 = Linear(1, d_model)
        self.mz_encoder1 = Linear(1, 100)
                                               
        self.encoder1 = TransformerEncoder(encoder_layer1, num_encoder_layers, encoder_norm)
                                             
                                                   
        self.encoder2 = TransformerEncoder(encoder_layer2, num_encoder_layers, encoder_norm)
        self.cross_atten = Cross_attention_Layer(d_model, nhead, dim_feedforward, 0, activation)
                                                   
                                                                                                      
        self.d_model = d_model
        self.max_len = max_len
        self.nhead = nhead
                                                                                                                                
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),                                             
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
                                                                      
                                             
                                                
                                                                        
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),                                              
            nn.Conv2d(10, 5, kernel_size=3, padding=1),                                             
            nn.BatchNorm2d(5),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),                                            
            nn.ReLU(inplace=True)
        )

                                             
                                           
                                                                                                                                              
        self.linear2 = Linear(5 * 24 * 124, 1000)                  
                                                                                              
                                                                       
        self.linear3 = Linear(1000 + 100, 100)
        self.linear4 = Linear(100, 23)
        self.linear5 = Linear(23, 1)

    def forward_encoder(self, src, src_mask=None, src_key_padding_mask=None, charge=None):
        charge = charge.unsqueeze(-1)
        charge = self.charge_encoder1(charge).unsqueeze(-1)
        charge = self.charge_encoder2(charge)
        src = self.linear1(src)
        src = src + charge
        src = src.transpose(0, 1)
                           
                         
        if src.size(2) != self.d_model:
            raise RuntimeError("the feature number of src must be equal to d_model")
        pep = self.encoder1(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        pep = self.encoder2(pep, mask=src_mask, src_key_padding_mask=None)
                                                                                      
        return pep

    def forward(self, pep1, pep2, src_mask=None, src_key_padding_mask_1=None, src_key_padding_mask_2=None, charge=None,
                mz=None) -> Tensor:
        pep1 = self.forward_encoder(pep1, src_mask, src_key_padding_mask_1, charge)
        pep2 = self.forward_encoder(pep2, src_mask, src_key_padding_mask_2, charge)
        mz = mz.unsqueeze(-1)
        mz = self.mz_encoder1(mz)
        pep = self.cross_atten(pep1, pep2)
        pep = pep.transpose(0, 1)                                                                                      
        pep = pep.unsqueeze(1)
        pep = self.conv(pep)
        batch = pep.size()[0]                    
        pep = pep.reshape(batch, -1)                      
        pep = self.linear2(pep)
        pep = F.relu(pep)
        pep = torch.cat((pep, mz), 1)
        pep = self.linear3(pep)
        pep = F.relu(pep)
                                     
        pep = self.linear4(pep)
        pep = F.relu(pep)
        ccs_pre = self.linear5(pep)
        return ccs_pre

                                                                   
     
                                                                      
                                                                                                      
                     

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class Cross_attention_Layer(Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(Cross_attention_Layer, self).__init__()
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
                                             
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(Cross_attention_Layer, self).__setstate__(state)

    def forward(self, pep1, pep2, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src1 = self.cross_attn(pep2, pep1, pep1, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src1 = pep1 + self.dropout1(src1)
        src1 = self.norm1(src1)

        src2 = self.cross_attn(pep1, pep2, pep2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = pep2 + self.dropout1(src2)
        src2 = self.norm1(src2)

        src3 = src1 + src2
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src3 = src3 + self.dropout2(src3)
        src3 = self.norm2(src3)

        return src3

class TransformerEncoder(Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
                                             
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def get_onehot_charge(charge):
    batch_size = charge.size(0)
    charge_m = torch.zeros(batch_size, 3).to(device='cuda', dtype=torch.float32)
    for i in range(batch_size):
        if charge[i] == 2:
            charge_m[i,] = torch.tensor([1, 0, 0]).to(device='cuda', dtype=torch.float32)
        elif charge[i] == 3:
            charge_m[i,] = torch.tensor([0, 1, 0]).to(device='cuda', dtype=torch.float32)
        elif charge[i] == 4:
            charge_m[i,] = torch.tensor([0, 0, 1]).to(device='cuda', dtype=torch.float32)
    return charge_m


def get_position_angle_vec(position, dim):                                   
    return [position / np.power(10000, 2 * (hid_j // 2) / dim) for hid_j in range(dim)]


def get_position_coding(max_len, d_model):
    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i, d_model) for pos_i in range(max_len)])                                    
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])                     
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])                      
    return sinusoid_table
