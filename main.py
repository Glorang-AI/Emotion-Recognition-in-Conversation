import os
import wandb
import argparse
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup,
    Wav2Vec2Config, 
    RobertaConfig, 
    BertConfig,
    AutoTokenizer
)

from sklearn.model_selection import train_test_split

from dataset import ETRIDataset
from trainer import ModelTrainer
from models import CASEmodel, RoCASEmodel, CompressedCCEModel, ConcatModel, MultiModalMixer
from utils import audio_embedding, seed

def main(args):
    seed.seed_setting(args.seed)
    
    # Pass the config dictionary when you initialize W&B
    wandb.init(project=args.wandb_project,
            group=args.wandb_group,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=args
    )

    wav_config = Wav2Vec2Config.from_pretrained(args.am_path)
    bert_config = BertConfig.from_pretrained(args.lm_path)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_path)

    def text_audio_collator(batch):
       
        return {'audio_emb' : pad_sequence([item['audio_emb'] for item in batch], batch_first=True),
                'label' : torch.stack([item['label'] for item in batch]).squeeze(),
                'input_ids' :  torch.stack([item['input_ids'] for item in batch]).squeeze(),
                'attention_mask' :  torch.stack([item['attention_mask'] for item in batch]).squeeze(),
                'token_type_ids' :  torch.stack([item['token_type_ids'] for item in batch]).squeeze()}

    dataset = pd.read_csv(args.data_path)
    dataset.reset_index(inplace=True)

    train_df, val_df = train_test_split(dataset, test_size = args.val_ratio, random_state=args.seed)

    # embedding path가 존재할 경우, 불러오며 없을 경우 생성한다.
    audio_emb = audio_embedding.save_and_load(args.am_path, dataset['audio'].tolist(), args.device, args.embedding_path)

    label_dict = {'angry':0, 'neutral':1, 'sad':2, 'happy':3, 'disqust':4, 'surprise':5, 'fear':6}
    train_dataset = ETRIDataset(
        audio_embedding = audio_emb, 
        dataset=train_df, 
        label_dict = label_dict,
        tokenizer = tokenizer,
        audio_emb_type = args.audio_emb_type,
        max_len = args.context_max_len, 
        )
    val_dataset = ETRIDataset(
        audio_embedding = audio_emb, 
        dataset=val_df, 
        label_dict = label_dict,
        tokenizer = tokenizer,
        audio_emb_type = args.audio_emb_type,
        max_len = args.context_max_len, 
        )

    # Create a DataLoader that batches audio sequences and pads them to a fixed length
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.train_bsz,
        shuffle=True, 
        collate_fn=text_audio_collator, 
        num_workers=args.num_workers,
        )
    valid_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.valid_bsz,
        shuffle=False, 
        collate_fn=text_audio_collator, 
        num_workers=args.num_workers,
        )

    loss_fn=nn.CrossEntropyLoss()

    if args.model == "CASE":
        model = CASEmodel(args.lm_path, wav_config, bert_config, args.num_labels)
    elif args.model == "CCE":
        model = CompressedCCEModel(args, wav_config, bert_config)
    elif args.model == "Concat":
        model = ConcatModel(args, wav_config, bert_config)
    elif args.model == "MMM":
        model = MultiModalMixer(args, wav_config, bert_config)
        model.freeze()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-5,
        no_deprecation_warning=True
        )
    
    scheduler = None
    if args.scheduler == "linear":
        total_steps = len(train_dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = total_steps * 0.1,
            num_training_steps = total_steps
        )

    contrastive_fn = None
    if args.contrastive:
        contrastive_fn = nn.BCEWithLogitsLoss()

    trainer = ModelTrainer(
        args,
        model, loss_fn, optimizer,
        train_dataloader, valid_dataloader,
        scheduler = scheduler,
        contrastive_loss_fn = contrastive_fn)

    trainer.train()
    
if __name__ == "__main__":

    # Define a config dictionary object
    parser = argparse.ArgumentParser()

    # -- Choose Pretrained Model
    parser.add_argument("--lm_path", type=str, default="klue/bert-base", help="You can choose models among (klue-bert series and klue-roberta series) (default: klue/bert-base")
    parser.add_argument("--am_path", type=str, default="kresnik/wav2vec2-large-xlsr-korean")

    # -- Training Argument
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train_bsz", type=int, default=64)
    parser.add_argument("--valid_bsz", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--context_max_len", type=int, default=128)
    parser.add_argument("--audio_max_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--num_labels", type=int, default=7)
    parser.add_argument("--audio_emb_type", type=str, default="last_hidden_state", help="Can chosse audio embedding type between 'last_hidden_state' and 'extract_features' (default: last_hidden_state)")
    parser.add_argument("--model", type=str, default="CASE")
    parser.add_argument("--contrastive", type=bool, default=False)

    ## -- directory
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--save_path", type=str, default="save")
    ###### emb_train에 대한 설명 부과하기
    parser.add_argument("--embedding_path", type=str, default="data/emb_train.pt")

    # -- utils
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    # -- wandb
    parser.add_argument("--wandb_project", type=str, default="comp")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default="case_audio_base")

    args = parser.parse_args()

    config = {
    "label_dict": {'angry':0, 'neutral':1, 'sad':2, 'happy':3, 'disqust':4, 'surprise':5, 'fear':6},
    "base_score":0.45, # Save the model according to the base validation score.
    }

    main(args)