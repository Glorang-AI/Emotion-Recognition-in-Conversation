import gc
import os
import torch
import soundfile as sf

from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

def save_and_load(audio_model_path, audio_data_path_data, device, save_path):
    
    if not os.path.isfile(save_path):
        
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_path)
        audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_path)
        audio_encoder.to(device)
        
        emb_dict=dict()
        
        with torch.no_grad():
            audio_encoder.eval()
            for idx, audio_path in enumerate(tqdm(audio_data_path_data)):
                sig, sr = sf.read(os.path.join("data", audio_path)) 
                
                audio_input = feature_extractor(sig, sampling_rate=sr, return_tensors='pt')['input_values'].to(device)
                outputs = audio_encoder(audio_input)
                
                emb_dict[str(idx)]={
                    'last_hidden_state':outputs['last_hidden_state'].squeeze().detach().cpu(),
                    'extract_features':outputs['extract_features'].squeeze().detach().cpu()
                    }
                
        torch.save(emb_dict, save_path)
        
        del audio_encoder, feature_extractor, audio_input, outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    else:
        emb_dict = torch.load(save_path)
    
    return emb_dict