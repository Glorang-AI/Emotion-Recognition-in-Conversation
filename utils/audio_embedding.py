import os
import torch
from tqdm import tqdm
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

def save_and_load(audio_model_path, audio_data_path_data, device, save_path):
    if not os.path.isfile(save_path):
        emb_dict=dict()
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_path)
        audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_path)
        audio_encoder.to(device)
        for idx, audio_path in enumerate(tqdm(audio_data_path_data)):
            sig, sr = sf.read(audio_path)
            audio_input = feature_extractor(sig, sampling_rate=sr, return_tensors='pt')['input_values'].to(device)
            outputs = audio_encoder(audio_input)
            
            emb_dict[str(idx)]={
                'last_hidden_state':outputs['last_hidden_state'].squeeze().detach().cpu(),
                'extract_features':outputs['extract_features'].squeeze().detach().cpu()
                }
            
        torch.save(emb_dict, save_path)
    
    else:
        emb_dict = torch.load(save_path)
    
    return emb_dict

# def label_to_tensor(data):
#     label_dict = {'angry':0, 'neutral':1, 'sad':2, 'happy':3, 'disqust':4, 'surprise':5, 'fear':6}
#     data['labels'] = label_dict[data['labels']]
#     return data