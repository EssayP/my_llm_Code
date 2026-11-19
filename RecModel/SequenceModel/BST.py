import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q = nn.Linear(d_model,d_model)
        self.k = nn.Linear(d_model,d_model)
        self.v = nn.Linear(d_model,d_model)
        self.output = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)

    def split_head(self,x):
        batch_size,seq_len,_ = x.shape
        x = x.view(batch_size,seq_len,self.num_heads,self.head_dim)
        return x.transpose(1,2)
    
    def forward(self,x,mask):
        q = self.split_head(self.q(x))
        k = self.split_head(self.k(x))
        v = self.split_head(self.v(x))

        attention_score = math.matmul(q,k.transpose(-1,-2)) / math.sqrt(self.head_dim)

        if mask is not None:
            attention_score = attention_score.masked_fill(~mask,-1e9)

        attention_weight = torch.softmax(attention_score,dim=-1)
        attention_weight = self.dropout(attention_weight)

        output = torch.matmul(attention_weight,v)
        output = output.transpose(1,2).contiguous().view(x.size(0),x.size(1),-1)

        output = self.output(output)

        return output, attention_weight
    

class TransformerBlock(nn.Module):
    def __init__(self,d_model.num_heads=8,ffn_dim=256,dropout=0.2):
        super().__init__()

        self.attention = MultiHeadSelfAttention(d_model,num_heads=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,ffn_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim,d_model),
        )

    def forward(self,x,mask):
        attention_output,attention_weight =self.attention(x,mask)
        x = self.norm1(x + self.dropout1(attention_output))

        ffn_output = self.ffn(x)

        x = self.norm2(x + self.dropout2(ffn_output))
        return x,attention_weight
    

class BST(nn.Module):
    def __init__(self,
                 num_items,
                 num_categories,
                 num_positions,
                 other_feature_dim,
                 item_dim=32,
                 category_dim=16,
                 position_dim=16,
                 num_heads=8,
                 num_blocks=1,
                 ffn_dim=256,
                 dropout=0.2):
        super().__init__()
        
        self.item_embedding = nn.Embedding(num_items +1,item_dim,padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories+1,category_dim,padding_idx=0)
        self.position_embedding = nn.Embedding(num_positions+1,position_dim,padding_idx=0)

        d_model = item_dim + category_dim + position_dim
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,num_heads,ffn_dim,dropout
                )
                for _ in range(num_blocks)
            ]
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model + other_feature_dim,1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,1),
        )

    def embed_embedding(self,item_ids,category_ids,position_ids):
        item_emb = self.category_embedding(item_ids)
        category_emb = self.category_embedding(category_ids)
        position_emb = self.category_embedding(position_ids)
        return torch.cat(item_emb,category_emb,position_emb,dim=-1)
    
    def forward(
            self,
            history_item_ids,
            history_category_ids,
            history_position,
            target_item_ids,
            target_category_ids,
            other_features,
    ):
        history_emb = self.embed_embedding(history_item_ids,history_category_ids,history_position,)
        target_position = torch.zeros_like(target_item_ids)
        target_emb = self.embed_embedding(
            target_item_ids,target_category_ids,target_position,
        )

        sequence = torch.cat([history_emb,target_emb],dim=1)

        history_mask = history_item_ids != 0
        target_mask = torch.ones(history_mask.size(0),1,dtype=torch.bool,device=history_mask.device)
        mask = torch.cat([history_mask,target_mask],dim=1)

        attention_weights = []
        for block in self.transformer_blocks:
            sequence,attention_weight = block(sequence,mask)
            attention_weights.append(attention_weight)

        target_state = sequence[:,-1,:]
        features = torch.cat([target_state,other_features],dim=-1)
        logits = self.mlp(features).squeeze(-1)

        return logits,attention_weights



