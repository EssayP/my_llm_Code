import torch
import torch.nn as nn

class Dice(nn.Module):
    def __init__(self,dim,eps=1e-9):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim,eps=eps)
        self.alpha = nn.Parameter(torch.zeros(dim))

    def forward(self,x):

        p = torch.sigmoid(self.bn(x))
        return self.alpha * (1-p) * x + p * x
    
class LocalActivateUnit(nn.Module):

    def __init__(self,item_dim,hidden_dims=(80,40)):
        super().__init__()

        layers=[]
        input_dim = item_dim * 4
        prev_dim = input_dim
        

        for hidden_dim in hidden_dims:
            layers.append(nn.Lnear(prev_dim,hidden_dim))
            layers.append(Dice(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim,1))
        self.mlp = nn.Sequential(*layers)

    def forward(self,query,keys,mask=None):
        """
        query 就是目前的candidate
        keys 就是用户历史行为序列的item
        计算candidate 和 历史序列里的每个item的 相关性
        """    

        batch_size,seq_len,item_dim = keys.shape()
        query = query.unsqueeze(1).expand(-1,keys.size(1),-1)
        input = torch.cat(
            [query,keys,query - keys, query * keys],dim = -1
        )
        
        input = input.reshape(batch_size * seq_len,-1)
        score = self.mlp(input)
        score = score.reshape(batch_size,seq_len)

        if mask is not None:
            score = score.masked_fill(~mask.bool(),float("-inf"))

        return torch.softmax(score,dim=-1)


class AttentionPooling(nn.Module):
    def __init__(self,embed_dim,hidden_dims=[80,40]):
        super().__init__()

        self.activation_unit = LocalActivateUnit(embed_dim,hidden_dims)

    def forward(self,query,keys,mask=None):
        attention_weights = self.activation_unit(query,keys,mask)
        attention_weights = attention_weights.unsqueeze(1)
        #使用计算得到的相关性进行加权求和
        interest = torch.sum(
            attention_weights * keys,
            dim = 1,
        )

        return interest
    

class DIN(nn.Module):
    def __init__(
            self,
            num_users,
            num_items,
            num_categories,
            embedding_dim=32,
            attention_hidden_units=(80,40),
            dnn_hidden_units=(200,80),
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim,
        )

        self.item_embdding = nn.Embeding(
            num_items,
            embedding_dim,
        )

        self.category_embedding = nn.Embedding(
            num_categories,
            embedding_dim,
        )

        behavior_dim = embedding_dim * 2

        self.attention_pooling = AttentionPooling(
            embedding_dim = behavior_dim,
            hidden_dims=attention_hidden_units,
        )

        dnn_input_dim = (
            embedding_dim              # user
            + behavior_dim             # target item + category
            + behavior_dim             # interest
        )

        layers = []

        prev_dim = dnn_input_dim

        for hidden_dim in dnn_hidden_units:

            layers.append(
                nn.Linear(
                    prev_dim,
                    hidden_dim,
                )
            )

            layers.append(nn.PReLU())

            prev_dim = hidden_dim

        layers.append(
            nn.Linear(prev_dim, 1)
        )

        self.dnn = nn.Sequential(*layers)

    def forward(
        self,
        user_ids,
        target_item_ids,
        target_category_ids,
        history_item_ids,
        history_category_ids,
        history_mask,
    ):


        user_emb = self.user_embedding(
            user_ids
        )


        target_item_emb = self.item_embedding(
            target_item_ids
        )

        target_category_emb = self.category_embedding(
            target_category_ids
        )

        # [B, 2D]
        target_emb = torch.cat(
            [
                target_item_emb,
                target_category_emb,
            ],
            dim=-1,
        )

        history_item_emb = self.item_embedding(
            history_item_ids
        )

        history_category_emb = self.category_embedding(
            history_category_ids
        )

        # [B, T, 2D]
        history_emb = torch.cat(
            [
                history_item_emb,
                history_category_emb,
            ],
            dim=-1,
        )

        # ============================================================
        # 4. DIN Attention
        #
        # Target Item -> Query
        # History Behaviors -> Keys
        # ============================================================

        interest_emb = self.attention_pooling(
            query=target_emb,
            behavior_embeddings=history_emb,
            mask=history_mask,
        )

        # [B, 2D]

        # ============================================================
        # 5. Concatenate
        # ============================================================

        dnn_input = torch.cat(
            [
                user_emb,
                target_emb,
                interest_emb,
            ],
            dim=-1,
        )

        # ============================================================
        # 6. Final DNN
        # ============================================================

        logits = self.dnn(
            dnn_input
        )

        logits = logits.squeeze(-1)

        return logits   
                             
                            
