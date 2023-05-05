import os
import datetime
import pandas as pd
import pytorch_lightning as pl
from typing import OrderedDict, Tuple
from transformers import AdamW
from src.model.bertsum import *
from src.model.utils import *
from src.utils.lr_scheduler import *
from src.rouge.rouge_score import RougeScorer


class ExtSum_Engine(pl.LightningModule):

    __doc__ = r"""
        pl-based engine for training a extractive summarization model.
        Unlike the english benchmark datasets(CNN/DM etc.), it has human-written extractive labels,
        so we evaluate the model with the given extractive labels instead of the abstractive. 
    
        Args:
            model: model instance to train
            train_df: train dataset in pd.DataFrame
            val_df: validation dataset in pd.DataFrame
            test_df: test dataset in pd.DataFrame
            sum_size: # sentences in a model-predicted summary
            n_block: n-gram size for n-gram blocking 
            model_checkpoint: checkpoints of only model
            freeze_base: freeze the model parameters while training
            lr: learning rate
            betas: betas of torch.optim.Adam
            weight_decay: weight_decay of torch.optim.Adam
            adam_epsilon: eps of torch.optim.Adam
            num_warmup_steps: # warm-up steps 
            num_training_steps: # total training steps
            save_result: save test result
             
        train_df, val_df and test_df must be given in order to get the candidate summary 
        from the prediction by indexing the document.
    """

    def __init__(
            self,
            model,
            train_df: Optional[pd.DataFrame] = None,
            val_df: Optional[pd.DataFrame] = None,
            test_df: Optional[pd.DataFrame] = None,
            sum_size: Optional[int] = 3,
            n_block: int = 3,
            model_checkpoint: Optional[str] = None,
            freeze_base: bool = False,
            lr: float = None,
            betas: Tuple[float] = (0.9, 0.999),
            weight_decay: float = 0.01,
            adam_epsilon: float = 1e-8,
            num_warmup_steps: int = None,
            num_training_steps: int = None,
            save_result: bool = False,
    ):
        super().__init__()

        self.model = model
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.sum_size = sum_size
        self.n_block = n_block

        # hparmas
        self.model_checkpoint = model_checkpoint
        self.freeze_base = freeze_base
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.save_result = save_result

        self.scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        self.prepare_training()


    def prepare_training(self):
        self.model.train()

        if self.model_checkpoint:
            checkpoint = torch.load(self.model_checkpoint)
            assert isinstance(checkpoint, OrderedDict), 'please load lightning-format checkpoints'
            assert next(iter(checkpoint)).split('.')[0] != 'model', 'this is only for loading the model checkpoints'
            self.model.load_state_dict(checkpoint)

        if self.freeze_base:
            for p in self.model.base_model.parameters():
                p.requires_grad = False


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        optim_params = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optim_params, self.lr, betas=self.betas, eps=self.adam_epsilon)
        scheduler = get_transformer_scheduler(optimizer, self.num_warmup_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {'scheduler': scheduler, 'interval': 'step'}
        }


    def training_step(self, batch, batch_idx):
        outputs = self.model(
            batch['encodings'],
            batch['cls_token_ids'],
            batch['ext_label'],
        )
        loss = outputs['loss']
        self.log('train_step_loss', loss, prog_bar=True)
        return {'loss': loss}


    def training_epoch_end(self, train_steps):
        losses = []
        for output in train_steps:
            losses.append(output['loss'])

        loss = sum(losses) / len(losses)
        self.log('train_loss', loss, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            batch['encodings'],
            batch['cls_token_ids'],
            batch['ext_label'],
        )
        loss = outputs['loss']
        preds = outputs['prediction']

        ref_sums, can_sums = [], []
        for i, id in enumerate(batch['id']):
            sample = self.val_df[self.val_df['id'] == id].squeeze()
            text = sample['text']

            ref_sum = [text[i] for i in sample['extractive']]
            ref_sums.append('\n'.join(ref_sum))

            can_sum = get_candidate_sum(text, preds[i], self.sum_size, self.n_block)
            can_sums.append('\n'.join(can_sum))

        return loss, ref_sums, can_sums


    def validation_epoch_end(self, val_steps):
        losses = []
        r1, r2, rL = [], [], []

        print('calculating ROUGE score...')
        for loss, ref_sums, can_sums in val_steps:
            for ref_sum, can_sum in zip(ref_sums, can_sums):
                rouge = self.scorer.score(ref_sum, can_sum)
                r1.append(rouge['rouge1'].fmeasure)
                r2.append(rouge['rouge2'].fmeasure)
                rL.append(rouge['rougeL'].fmeasure)

            losses.append(loss)

        loss = sum(losses) / len(losses)
        r1 = 100 * (sum(r1) / len(r1))
        r2 = 100 * (sum(r2) / len(r2))
        rL = 100 * (sum(rL) / len(rL))

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_rouge1', r1, prog_bar=True)
        self.log('val_rouge2', r2, prog_bar=True)
        self.log('val_rougeL', rL, prog_bar=True)


    def test_step(self, batch, batch_idx):
        outputs = self.model(
            batch['encodings'],
            batch['cls_token_ids'],
        )
        preds = outputs['prediction']

        texts, ref_sums, can_sums = [], [], []
        for i, id in enumerate(batch['id']):
            sample = self.test_df[self.test_df['id'] == id].squeeze()
            text = sample['text']
            texts.append('\n'.join(text))

            ref_sum = [text[i] for i in sample['extractive']]
            ref_sums.append('\n'.join(ref_sum))

            can_sum = get_candidate_sum(text, preds[i], self.sum_size, self.n_block)
            can_sums.append('\n'.join(can_sum))

        return texts, ref_sums, can_sums


    def test_epoch_end(self, test_steps):
        result = {
            'text': [],
            'reference summary': [],
            'candidate summary': [],
        }
        r1, r2, rL = [], [], []

        print('calculating ROUGE score...')
        for texts, ref_sums, can_sums in test_steps:
            for i, (ref_sum, can_sum) in enumerate(zip(ref_sums, can_sums)):
                rouge = self.scorer.score(ref_sum, can_sum)
                r1.append(rouge['rouge1'].fmeasure)
                r2.append(rouge['rouge2'].fmeasure)
                rL.append(rouge['rougeL'].fmeasure)

                if self.save_result:
                    result['text'].append(texts[i])
                    result['reference summary'].append(ref_sum)
                    result['candidate summary'].append(can_sum)

        r1 = 100 * (sum(r1) / len(r1))
        r2 = 100 * (sum(r2) / len(r2))
        rL = 100 * (sum(rL) / len(rL))

        self.log('test_rouge1', r1, prog_bar=True)
        self.log('test_rouge2', r2, prog_bar=True)
        self.log('test_rougeL', rL, prog_bar=True)

        if self.save_result:
            path = 'result/{}'.format(datetime.datetime.now().strftime('%y-%m-%d'))
            if not os.path.exists(path):
                os.makedirs(path)

            result_pd = pd.DataFrame(result)
            result_pd.to_csv(path + '/{}.csv'.format(datetime.datetime.now().strftime('%H-%M-%S')), index=False)