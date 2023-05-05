import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel, BatchEncoding
from .encoder import SumEncoder

Tensor = torch.Tensor


class BertSum_Ext(nn.Module):

    __doc__ = r"""
        Implementation of the paper; 
        https://arxiv.org/pdf/1908.08345.pdf
    """

    def __init__(
            self,
            base_checkpoint: str,
            enc_num_layers: int = 2,
            enc_intermediate_size: int = 2048,
            enc_num_attention_heads: int = 8,
            enc_dropout_prob: float = 0.2,
    ):
        super().__init__()

        self.base_checkpoint = base_checkpoint
        self.base_model = AutoModel.from_pretrained(self.base_checkpoint)

        enc_hidden_size = self.base_model.config.hidden_size

        self.head = SumEncoder(
            enc_num_layers,
            enc_hidden_size,
            enc_intermediate_size,
            enc_num_attention_heads,
            enc_dropout_prob,
        ).eval()

        self.loss_fn = nn.BCELoss(reduction='none')


    def forward(
            self,
            encodings: BatchEncoding,
            cls_token_ids: Tensor,
            ext_labels: Optional[Tensor] = None,
    ):
        token_embeds = self.base_model(**encodings).last_hidden_state
        _, cls_mask, cls_logits = self.head(token_embeds, cls_token_ids).values()

        scores = torch.sigmoid(cls_logits) * cls_mask
        num_sents = torch.sum(cls_mask, dim=-1)

        loss = None
        if not (self.loss_fn is None or ext_labels is None):
            loss = self.loss_fn(scores, ext_labels.float())
            loss = (loss * cls_mask).sum() / num_sents.sum()

        prediction, confidence = [], []
        for i, score in enumerate(scores):
            conf, pred = torch.sort(score[cls_mask[i] == 1], descending=True, dim=-1)
            prediction.append(pred.tolist())
            confidence.append(conf)

        return {
            'logits': cls_logits,
            'loss': loss,
            'prediction': prediction,
            'confidence': confidence,
        }
