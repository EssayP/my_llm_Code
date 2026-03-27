import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class ScaledDotPorductAttention(nn.Module):
    def __init__(self,dropout:float = 0.1):
        super(ScaledDotPorductAttention,self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                Q:torch.Tensor,
                K:torch.Tensor,
                V:torch.Tensor,
                mask:torch.Tensor = None
        )-> tuple[torch.Tensor,torch.Tensor]:
        d_k = Q.size(-1)
        scores = torch.matmul(Q,K.transpose(-2,-1)) / (math.sqrt(d_k))

        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            # 把不允许关注的位置置为负无穷，使得在softmax后这些位置的权重变诶0
            scores=scores.masked_fill(~mask,float('-inf'))

        attn_weights = F.softmax(scores,dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights,V)
        return output,attn_weights
