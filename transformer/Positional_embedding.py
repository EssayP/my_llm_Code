from tkinter import Variable

import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)  # 初始化dropout层

        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(1) #[[0],[1],[2],...] max_len * 1
        # 指数和对数运算的原因是为了确保数值稳定性和计算效率
        div_term = torch.exp(torch.arange(0,d_model,2) *
                             -(math.log(10000.0)/d_model))
        # max_len * d_model/2
        # arr[start:end:step]
        pe[:,0::2] = torch.sin(position * div_term) #从0开始，步幅为2
        pe[:,1::2] = torch.cos(position * div_term) #从1 开始，步幅为2
        pe = pe.unsqueeze(0) #(1 x max_len x d_model) 方便进行批处理
        self.register_buffer('pe',pe)

    def forward(self,x):
        # 输入x 的维度是(batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]  # 自动广播
        return self.dropout(x)