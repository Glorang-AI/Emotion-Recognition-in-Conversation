import random
import numpy as np
import torch
import pandas as pd
from typing import Callable, Tuple, Dict
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class CASEDataset(Dataset):
    """
    Audio and Text data loader
    """
    def __init__(self, 
                 dataset:pd.DataFrame, 
                 label_dict:Dict, # label to idx dictionary
                 tokenizer:Callable,
                 max_len:int=128):
        super().__init__()
        self.dataset = dataset
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.label_dict = label_dict
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        self.dataset.iloc['text'][idx]

token = tokenizer.batch_encode_plus(text, return_tensors='pt', \
                            add_special_tokens=True, \
                            max_length=128, \
                            padding='max_length', \
                            truncation=True, \
                            return_attention_mask=True, \
                            return_token_type_ids=True)
    