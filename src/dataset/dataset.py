import torch
import pandas as pd
from typing import Union
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ExtSum_Dataset(Dataset):

    __doc__ = r"""
        Referred to the repos below;
        https://github.com/nlpyang/PreSumm
        https://github.com/KPFBERT/kpfbertsum

        Returns:
            ids: 'id' value of the data, which is to index the document from the prediction
            encodings: input_ids, token_type_ids, attention_mask
                token_type_ids alternates between 0 and 1 to separate the sentences
            cls_token_ids: identify CLS tokens representing sentences among all tokens
            ext_label: extractive label to train in sentence-level binary classification
    """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]

        # load and tokenize each sentence
        encodings = []
        for sent in row['text']:
            encoding = self.tokenizer(
                sent,
                add_special_tokens=True,
            )
            encodings.append(encoding)

        input_ids, token_type_ids, attention_mask = [], [], []
        ext_label, cls_token_ids = [], []

        # seperate each of sequences
        seq_id = 0
        for enc in encodings:
            if seq_id > 1:
                seq_id = 0
            cls_token_ids += [len(input_ids)]
            input_ids += enc['input_ids']
            token_type_ids += len(enc['input_ids']) * [seq_id]
            attention_mask += len(enc['input_ids']) * [1]

            if encodings.index(enc) in row['extractive']:
                ext_label += [1]
            else:
                ext_label += [0]

            seq_id += 1

        # pad and truncate inputs
            if len(input_ids) == self.max_seq_len:
                break

            elif len(input_ids) > self.max_seq_len:
                sep = input_ids[-1]
                input_ids = input_ids[:self.max_seq_len - 1] + [sep]
                token_type_ids = token_type_ids[:self.max_seq_len]
                attention_mask = attention_mask[:self.max_seq_len]
                break

        if len(input_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids += pad_len * [self.pad]
            token_type_ids += pad_len * [0]
            attention_mask += pad_len * [0]

        # adjust for BertSum_Ext
        if len(cls_token_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(cls_token_ids)
            cls_token_ids += pad_len * [-1]
            ext_label += pad_len * [0]

        encodings = BatchEncoding(
            {
                'input_ids': torch.tensor(input_ids).to(device),
                'token_type_ids': torch.tensor(token_type_ids).to(device),
                'attention_mask': torch.tensor(attention_mask).to(device),
            }
        )
        return dict(
            id=row['id'],
            encodings=encodings,
            cls_token_ids=torch.tensor(cls_token_ids).to(device),
            ext_label=torch.tensor(ext_label).to(device)
        )
