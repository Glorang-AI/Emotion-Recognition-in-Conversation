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
  'data_path' : 'data/train.csv',
  "label_dict": {'angry':0, 'neutral':1, 'sad':2, 'happy':3, 'disqust':4, 'surprise':5, 'fear':6},
  "sav_dir":'save',
  "base_score":0.45, # Save the model according to the base validation score.
  'embedding_path':'data/emb_train.pt', # If an embedding file named "data/emb_train.pt" does not exist, generate one
  "audio_emb_type": 'last_hidden_state', # audio embedding type: 'last_hidden_state' or 'extract_features'
  "max_len" : 128,
  "seed":42
}


seed.seed_setting(config['seed'])

# # Pass the config dictionary when you initialize W&B
# wandb.init(project='comp',
#         group='bert_cls',
#         name='case_audio_base',
#         config=config
# )

wav_config = Wav2Vec2Config.from_pretrained(config['am_path'])
bert_config = BertConfig.from_pretrained(config['lm_path'])
tokenizer = AutoTokenizer.from_pretrained(config['lm_path'])

def text_audio_collator(batch):
    audio_emb = pad_sequence([item['audio_emb'] for item in batch], batch_first=True)
    batch['audio_emb'] = audio_emb
    return batch

dataset = pd.read_csv(config['data_path'])
dataset.reset_index(inplace=True)
train_df, val_df = train_test_split(dataset, test_size = config['val_ratio'], random_state=config['seed'])

audio_emb = audio_embedding.save_and_load(config['am_path'], dataset['audio'].to_list(),
                                                'cuda:3',  # cuda is required to run the audio embedding generation model.
                                                config['embedding_path']) 

train_dataset = ETRIDataset(audio_embedding = audio_emb, 
                                dataset=train_df, 
                                label_dict = config['label_dict'],
                                tokenizer = tokenizer,
                                audio_emb_type = config['audio_emb_type'],
                                max_len = config['max_len'], 
                                )

for i in train_dataset:
    print(i['label'])