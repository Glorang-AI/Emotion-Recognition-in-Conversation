import torch
import wandb
import pandas as pd
from torch import nn
from dataset import ETRIDataset
from trainer import ModelTrainer
from torch.utils.data import DataLoader
from utils import audio_embedding, seed
from models import CASEmodel, RoCASEmodel
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from transformers import AdamW, Wav2Vec2Config, RobertaConfig, BertConfig, AutoTokenizer

# Define a config dictionary object
config = {
  "lr": 1e-5,
  "lm_path":'klue/bert-base',
  "am_path": 'kresnik/wav2vec2-large-xlsr-korean',
  "train_bsz": 64,
  "val_bsz": 64,
  "val_ratio":0.1,
  "max_len": 128,
  "epochs" :20,
  "device":'cuda:2',
  "num_labels":7,
  "num_workers":4,
  'data_path' : 'data/train.csv',
  "label_dict": {'angry':0, 'neutral':1, 'sad':2, 'happy':3, 'disqust':4, 'surprise':5, 'fear':6},
  "sav_dir":'save',
  "base_score":0.45, # Save the model according to the base validation score.
  'embedding_path':'data/emb_train.pt', # If an embedding file named "data/emb_train.pt" does not exist, generate one
  "audio_emb_type": 'last_hidden_state', # audio embedding type: 'last_hidden_state' or 'extract_features'
  "max_len" : 128,
  "seed":42
}

def main():
    seed.seed_setting(config['seed'])
    
    # Pass the config dictionary when you initialize W&B
    wandb.init(project='comp',
            group='bert_cls',
            name='case_audio_base',
            config=config
    )

    wav_config = Wav2Vec2Config.from_pretrained(wandb.config['am_path'])
    bert_config = BertConfig.from_pretrained(wandb.config['lm_path'])
    tokenizer = AutoTokenizer.from_pretrained(wandb.config['lm_path'])

    def text_audio_collator(batch):
       
        
        return {'audio_emb' : pad_sequence([item['audio_emb'] for item in batch], batch_first=True),
                'label' : torch.stack([item['label'] for item in batch]).squeeze(),
                'input_ids' :  torch.stack([item['input_ids'] for item in batch]).squeeze(),
                'attention_mask' :  torch.stack([item['attention_mask'] for item in batch]).squeeze(),
                'token_type_ids' :  torch.stack([item['token_type_ids'] for item in batch]).squeeze()}
                
        # audio_emb = pad_sequence([item.pop('input_ids') for item in batch], batch_first=True)
        return {'text_encoded':batch, 'audio_emb':audio_emb}
        return {"label": label, "text_input": token, "audio_emb":audio_emb}

    dataset = pd.read_csv(wandb.config['data_path'])
    dataset.reset_index(inplace=True)
    train_df, val_df = train_test_split(dataset, test_size = wandb.config['val_ratio'], random_state=config['seed'])

    audio_emb = audio_embedding.save_and_load(wandb.config['am_path'], dataset['audio'].to_list(),
                                                    'cuda:3',  # cuda is required to run the audio embedding generation model.
                                                    wandb.config['embedding_path']) 

    train_dataset = ETRIDataset(audio_embedding = audio_emb, 
                                dataset=train_df, 
                                label_dict = wandb.config['label_dict'],
                                tokenizer = tokenizer,
                                audio_emb_type = wandb.config['audio_emb_type'],
                                max_len = wandb.config['max_len'], 
                                )

    val_dataset = ETRIDataset(audio_embedding = audio_emb, 
                            dataset=val_df, 
                            label_dict = wandb.config['label_dict'],
                            tokenizer = tokenizer,
                            audio_emb_type = wandb.config['audio_emb_type'],
                            max_len = wandb.config['max_len'], 
                            )

    # Create a DataLoader that batches audio sequences and pads them to a fixed length
    train_dataloader = DataLoader(train_dataset, batch_size=wandb.config['train_bsz'],
                                shuffle=True, collate_fn=text_audio_collator, num_workers=wandb.config['num_workers'])
    valid_dataloader = DataLoader(val_dataset, batch_size=wandb.config['val_bsz'],
                                shuffle=False, collate_fn=text_audio_collator, num_workers=wandb.config['num_workers'])

    loss_fn=nn.CrossEntropyLoss()
    model = CASEmodel(wandb.config['lm_path'], wav_config, bert_config, wandb.config['num_labels'])
    optimizer = AdamW(model.parameters(),
                        lr=1e-5,
                        no_deprecation_warning=True)
    wandb.config["num_labels"]

    trainer = ModelTrainer(model, loss_fn, optimizer, wandb.config["device"], 
                wandb.config["sav_dir"], train_dataloader, valid_dataloader, 
                wandb.config["epochs"], wandb.config["base_score"], wandb.config["num_labels"])

    trainer.train()
    
if __name__ == "__main__":
    main()