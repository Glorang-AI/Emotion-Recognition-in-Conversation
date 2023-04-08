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
    BertConfig,
    AutoTokenizer
)

from sklearn.model_selection import train_test_split

from dataset import ETRIDataset
from trainer import ModelTrainer
from models import (
    CASEAttentionModel, 
    CASECompressingModel,
    ConcatModel, 
    MultiModalMixer,
    TextOnlyModel,
    SpeechOnlyModel,
)
from utils import audio_embedding, seed, loss

def main(args):

    # seed number setting
    seed.seed_setting(args.seed)
    
    # Pass the config dictionary when you initialize W&B
    if args.mode == "train":
        wandb.init(project=args.wandb_project,
                group=args.wandb_group,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config=args
        )

    def text_audio_collator(batch):
        """
            # Create a DataLoader that batches audio sequences and pads them to a fixed length
        """
        return {'audio_emb' : pad_sequence([item['audio_emb'] for item in batch], batch_first=True),
                'label' : torch.stack([item['label'] for item in batch]).squeeze(),
                'input_ids' :  torch.stack([item['input_ids'] for item in batch]).squeeze(),
                'attention_mask' :  torch.stack([item['attention_mask'] for item in batch]).squeeze(),
                'token_type_ids' :  torch.stack([item['token_type_ids'] for item in batch]).squeeze()}

    # label 변환
    label_dict = {'angry':0, 'neutral':1, 'sad':2, 'happy':3, 'disqust':4, 'surprise':5, 'fear':6}
    pet_label_dict = {'angry':'분노', 'neutral':'중립', 'sad':'슬픔', 'happy':'행복', 'disqust':'불쾌', 'surprise':'경이', 'fear':'공포'}

    # 각 Modality별 사전학습 모델의 config 반환
    wav_config = Wav2Vec2Config.from_pretrained(args.am_path)
    bert_config = BertConfig.from_pretrained(args.lm_path)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_path)

    # -- Model Setting
    if args.model == "attention":
        model = CASEAttentionModel(args, wav_config, bert_config)
    elif args.model == "compressing":
        model = CASECompressingModel(args, wav_config, bert_config)
    elif args.model == "Concat":
        model = ConcatModel(args, wav_config, bert_config)
    elif args.model == "MMM":
        model = MultiModalMixer(args, wav_config, bert_config)
        # model.freeze()
    elif args.model == "text_only":
        model = TextOnlyModel(args, bert_config)
    elif args.model == "speech_only":
        model = SpeechOnlyModel(args, wav_config)
    
    # Test 수행
    if args.mode == "test":
        model.load_state_dict(torch.load(args.test_model_path))

        test_data = pd.read_csv(args.test_path)
        test_data.reset_index(inplace=True)

        test_audio_emb = audio_embedding.save_and_load(args.am_path, test_data['audio'].tolist(), args.device, args.test_embedding_path)

        test_dataset = ETRIDataset(
            audio_embedding = test_audio_emb, 
            dataset=test_data, 
            label_dict = label_dict,
            tokenizer = tokenizer,
            audio_emb_type = args.audio_emb_type,
            max_len = args.context_max_len, 
            pet=args.pet
            )
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.valid_bsz,
            shuffle=False, 
            collate_fn=text_audio_collator, 
            num_workers=args.num_workers,
            )
        
        trainer = ModelTrainer(
            args,
            model, loss_fn=None, optimizer=None, tokenizer=tokenizer,
            train_dataloader=None, valid_dataloader=None, test_dataloader=test_dataloader,
            scheduler = None,
            verbalizer_value=pet_label_dict if args.pet else None,
            label_dict = label_dict
            )

        trainer.test()
        return
    
    # 이하 Train 수행

    dataset = pd.read_csv(args.train_path)
    dataset.reset_index(inplace=True)

    # embedding path가 존재할 경우, 불러오며 없을 경우 생성한다.
    audio_emb = audio_embedding.save_and_load(args.am_path, dataset['audio'].tolist(), args.device, args.embedding_path)

    # Train Dataset을 Train-Valid Dataset으로 나눈다.
    if args.val_ratio != 0:
        train_df, val_df = train_test_split(dataset, test_size = args.val_ratio, random_state=args.seed)
        
        train_dataset = ETRIDataset(
            audio_embedding = audio_emb, 
            dataset=train_df, 
            label_dict = label_dict,
            tokenizer = tokenizer,
            audio_emb_type = args.audio_emb_type,
            max_len = args.context_max_len, 
            pet=args.pet
            )
        
        val_dataset = ETRIDataset(
            audio_embedding = audio_emb, 
            dataset=val_df, 
            label_dict = label_dict,
            tokenizer = tokenizer,
            audio_emb_type = args.audio_emb_type,
            max_len = args.context_max_len, 
            pet=args.pet
            )

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

    else:
        train_dataset = ETRIDataset(
            audio_embedding = audio_emb, 
            dataset=dataset, 
            label_dict = label_dict,
            tokenizer = tokenizer,
            audio_emb_type = args.audio_emb_type,
            max_len = args.context_max_len, 
            pet=args.pet
            )

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.train_bsz,
            shuffle=True, 
            collate_fn=text_audio_collator, 
            num_workers=args.num_workers,
            )
        
        valid_dataloader = None
    
    # -- Loss Setting
    if args.loss == "focal":
        loss_fn = loss.FocalLoss(gamma = args.gamma)
    else:
        loss_fn=nn.CrossEntropyLoss()
        
    # -- Optimizer Setting
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        no_deprecation_warning=True
        )
    
    # -- Scheduler Setting
    scheduler = None
    if args.scheduler == "linear":
        total_steps = len(train_dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = total_steps * 0.1,
            num_training_steps = total_steps
        )

    trainer = ModelTrainer(
        args,
        model, loss_fn, optimizer, tokenizer,
        train_dataloader, valid_dataloader, test_dataloader=None,
        scheduler = scheduler,
        verbalizer_value=pet_label_dict if args.pet else None,
        label_dict = label_dict
        )
    trainer.train()
    
if __name__ == "__main__":

    # Define a config dictionary object
    parser = argparse.ArgumentParser()

    # -- Choose Pretrained Model
    parser.add_argument("--lm_path", type=str, default="klue/bert-base", help="You can choose models among (klue-bert series and klue-roberta series) (default: klue/bert-base")
    parser.add_argument("--am_path", type=str, default="kresnik/wav2vec2-large-xlsr-korean")

    # -- Training Argument
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_bsz", type=int, default=64)
    parser.add_argument("--valid_bsz", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--context_max_len", type=int, default=128)
    parser.add_argument("--audio_max_len", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--pet", type=bool, default=False)
    parser.add_argument("--loss", type=str, default="crossentropy")
    parser.add_argument("--gamma", type=float, default=1.0, help="focalloss's gamma argument")
    
    # -- Model Argument
    parser.add_argument("--model", type=str, default="CASE")
    parser.add_argument("--audio_emb_type", type=str, default="last_hidden_state", help="Can choose audio embedding type between 'last_hidden_state' and 'extract_features' (default: last_hidden_state)")
    parser.add_argument("--opt", type=str, default='mean', help="Can choose operators type between 'mean' and 'sum'")
    parser.add_argument("--mm_type", type=str, default='add', help="concat or add")
    parser.add_argument("--num_labels", type=int, default=7)

    ## -- directory
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--test_path", type=str, default="data/test.csv")
    parser.add_argument("--save_path", type=str, default="save")
    parser.add_argument("--embedding_path", type=str, default="data/emb_train.pt")
    parser.add_argument("--test_embedding_path", type=str, default="data/emb_test.pt")
    parser.add_argument("--test_model_path", type=str, default="save/e150_compressing_seed0.pt")

    # -- utils
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    # -- wandb
    parser.add_argument("--wandb_project", type=str, default="comp")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default="case_audio_base")
    
    ## -- mode
    parser.add_argument("--mode", type=str, default="train")

    args = parser.parse_args()

    main(args)
    
