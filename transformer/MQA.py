import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class MultiQueryAttention(nn.Module):
    def __init__(self,embed_dim,head_nums,dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_nums = head_nums
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // head_nums

        self.wq = nn.Linear(embed_dim,embed_dim)
        self.wk = nn.Linear(embed_dim,self.head_dim)
        self.wv = nn.Linear(embed_dim,self.head_dim)

        self.wo = nn.Linear(embed_dim,embed_dim)

    def forward(self,x,mask=None):
        batch_size,seq_len,_ = x.size()
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(batch_size, seq_len, self.head_nums, self.head_dim).transpose(1,2)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        scores = torch.matmul(q,k.transpose(-2,-1)) / (self.head_dim ** 0.5)

        if mask is not None:
            mask = mask.to(torch.bool)
            scores = scores.masked_fill(~mask,float('-inf'))

        attn_weights = F.softmax(scores,dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights,v)
        out = out.transpose(1,2).contiguous().view(batch_size,seq_len,self.embed_dim)

        out = self.wo(out)

        return out



