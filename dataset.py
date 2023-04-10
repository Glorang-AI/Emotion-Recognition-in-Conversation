import torch
import pandas as pd

from typing import Callable, Dict
from torch.utils.data import Dataset

class ETRIDataset(Dataset):
    """
    This is a class that returns audio embeddings and text tokenization results.
    """
    def __init__(self, 
                 audio_embedding, # audio embedding load from '~.pt' file
                 dataset:pd.DataFrame, 
                 label_dict:Dict, # label to idx dictionary
                 tokenizer:Callable,
                 audio_emb_type:str='last_hidden_state', # audio embedding type: 'last_hidden_state' or 'extract_features'
                 max_len:int=128,
                 pet:bool=False):
        super().__init__()
        
        self.audio_emb = audio_embedding
        self.dataset = dataset
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.emb_type = audio_emb_type
        self.pet = pet
        
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset['text'].iloc[idx]
        
        if self.pet:
            text='다음 문장의 감정은 [MASK]입니다.[SEP]'+text # pattern(prompt)
            
        label = torch.tensor(self.label_dict[self.dataset['labels'].iloc[idx]])
        
        emb_key = str(self.dataset.index[idx])
        wav_emb = self.audio_emb[emb_key][self.emb_type]

        encoded_dict = self.tokenizer(text, 
                                      return_tensors='pt',
                                      add_special_tokens=True,
                                      max_length=self.max_len,
                                      padding='max_length',
                                      truncation=True,
                                      return_attention_mask=True,
                                      return_token_type_ids=True
                                      )
        
        encoded_dict['audio_emb'] = wav_emb
        encoded_dict['label'] = label
        
        return encoded_dict
