import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,head_num,dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.head_dim = embed_dim // head_num
        self.dropout = nn.Dropout(dropout)

        self.W_Q = nn.Linear(embed_dim,embed_dim)
        self.W_K = nn.Linear(embed_dim,embed_dim)
        self.W_V = nn.Linear(embed_dim,embed_dim)
        self.W_O = nn.Linear(embed_dim,embed_dim)

    def forward(self,x,mask):
        # x:[batch_size,seq_len,embed_dim]
        batch_size,seq_len,embed_dim = x.size()
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
        K = K.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
        V = V.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)

        score = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.head_dim)

        # 处理padding
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            score = score.masked_fill(mask ==0,float('inf'))

        attn_weights = F.softmax(score,dim=-1)
        attn_weights = self.dropout(attn_weights)
        output =torch.matmul(attn_weights,V)

        output = output.transpose(1,2).contiguous().view(batch_size,seq_len,embed_dim)
        output = self.W_O(output)
        return output


if __name__ == '__main__':
    embed_size = 512
    head_num = 8
    batch_size = 64
    seq_len = 10
    dropout = 0.1
    mha = MultiHeadAttention(embed_size,head_num,dropout)
    x = torch.randn(batch_size,seq_len,embed_size)
    print(mha.forward(x,mask=None).shape)