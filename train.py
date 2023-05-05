import argparse
import hydra
from src import *

parser = argparse.ArgumentParser()
parser.add_argument("--config-name", dest='config_name', default=None, type=str)
args = parser.parse_args()


@hydra.main(version_base=None, config_path='./config/', config_name=args.config_name)
def train(cfg: DictConfig):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    # prepare tokenizer, model and loss function
    model = BertSum_Ext(**cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_checkpoint)

    # load train and validation datasets
    train_df, val_df = get_train_df(cfg.dataset.path, **cfg.dataset.df)

    train_dataset = ExtSum_Dataset(train_df, tokenizer, cfg.max_seq_len)
    val_dataset = ExtSum_Dataset(val_df, tokenizer, cfg.max_seq_len)
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=False)

    # config training
    engine = ExtSum_Engine(model, train_df, val_df, sum_size=3, n_block=3, **cfg.engine)
    logger = Another_WandbLogger(**cfg.log, save_artifact=False)
    cfg_trainer = Config_Trainer(cfg.trainer)()

    # run training
    trainer = pl.Trainer(
        **cfg_trainer,
        logger=logger,
        num_sanity_val_steps=0
    )
    logger.watch(engine)

    if cfg.train_checkpoint:
        trainer.fit(engine, train_loader, val_loader, ckpt_path=cfg.train_checkpoint)
    else:
        trainer.fit(engine, train_loader, val_loader)

    wandb.finish()



if __name__ == "__main__":
    train()
