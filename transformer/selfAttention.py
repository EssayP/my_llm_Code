import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self,embed_dim,d_k):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_k = d_k
        self.W_Q = nn.Linear(embed_dim,d_k)
        self.W_K = nn.Linear(embed_dim,d_k)
        self.W_V = nn.Linear(embed_dim,d_k)

    def forward(self,x):
        # x:[batch_size,seq_len,embed_dim]
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        score = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(score,dim=-1)
        attn_output = torch.matmul(attn_weights,V)
        return attn_output,attn_weights