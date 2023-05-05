import math
import torch
import torch.nn as nn
from transformers import BertLayer, BertConfig


class SumEncoder(nn.Module):

    __doc__ = r"""
        cls_attention_mask prevents padding tokens from being included in softmax values 
        inside the encoder's self-attention layer.
    """

    def __init__(
            self,
            num_layers: int,
            hidden_size: int,
            intermediate_size: int,
            num_attention_heads: int,
            dropout_prob: float,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob

        self.position_embedding = PositionEmbedding(dropout_prob, hidden_size)
        self.layers = nn.ModuleList([self.bert_layer() for _ in range(self.num_layers)])

        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, 1, bias=True)
        )


    def bert_layer(self):
        config = BertConfig()
        config.hidden_size = self.hidden_size
        config.intermediate_size = self.intermediate_size
        config.num_attention_heads = self.num_attention_heads
        config.attention_probs_dropout_prob = self.dropout_prob
        config.hidden_dropout_prob = self.dropout_prob

        return BertLayer(config)


    def cls_attention_mask(self, cls_token_mask):
        attention_mask = cls_token_mask[:, None, None, :]
        attention_mask = attention_mask.expand(
            -1, self.num_attention_heads, attention_mask.size(-1), -1)
        attention_mask = (1.0 - attention_mask) * -1e18

        return attention_mask


    def forward(self, last_hidden_state, cls_token_ids):
        cls_token_mask = (cls_token_ids != -1).float()
        cls_index = torch.arange(last_hidden_state.size(0)).unsqueeze(1), cls_token_ids
        cls_embed = last_hidden_state[cls_index]
        cls_embed = cls_embed * cls_token_mask[:, :, None]

        if self.num_layers:
            pos_embed = self.position_embedding.pe[:, :last_hidden_state.size(1)]
            cls_embed = cls_embed + pos_embed
            attention_mask = self.cls_attention_mask(cls_token_mask)

            for i in range(self.num_layers):
                cls_embed = self.layers[i](cls_embed, attention_mask=attention_mask)[0]
            cls_embed = cls_embed * cls_token_mask[:, :, None]

        logits = self.last_layer(cls_embed).squeeze(-1)
        logits = logits * cls_token_mask

        return {
            'cls_embeddings': cls_embed,
            'cls_token_mask': cls_token_mask,
            'logits': logits,
        }



class PositionEmbedding(nn.Module):

    def __init__(
            self,
            dropout_prob: float,
            dim: int,
            max_len: int = 5000
    ):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))

        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)

        super().__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim


    def get_embed(self, embed):
        return self.pe[:, :embed.size(1)]


    def forward(self, embed, step=None):
        embed = embed * math.sqrt(self.dim)
        if step:
            embed = embed + self.pe[:, step][:, None, :]
        else:
            embed = embed + self.pe[:, :embed.size(1)]
        embed = self.dropout(embed)
        return embed
