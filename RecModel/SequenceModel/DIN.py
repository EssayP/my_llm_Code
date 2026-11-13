import torch
import torch.nn as nn

class Dice(nn.Module):
    def __init__(self,input_dim,eps=1e-8):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim,eps=eps)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self,x):
        x = self.bn(x)
        p = torch.sigmoid(x)
        return p * x + (1-p) * self.alpha * x
    
class ActivationUnit(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_dims=(64,32),
                 dropout=0.0):
        super().__init__()

        input_dim = 4 * embed_dim
        layers: list[nn.Module] = []
        for hidden_dim in hidden_dims:
            layer.append(
                [
                    nn.Linear(input_dim,hidden_dim),
                    Dice(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim,1))
        self.network = nn.Sequential(*layers)

    def forward(self, query,user_behavior):
        seq_len = user_behavior.size(1)
        queries = query.expand(-1,seq_len,-1)
        attention_input = torch,cat(
            [queries,user_behavior,queries-user_behavior,queries*user_behavior],dim = -1
        )
        return self.network(attention_input).sequeeze(-1)



