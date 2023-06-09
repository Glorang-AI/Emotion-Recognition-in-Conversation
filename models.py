import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers import BertPreTrainedModel, BertModel

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
        
class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class CASEAttentionModel(BertPreTrainedModel):
    """
        CASE (attention)
    """
    def __init__(self, args, wav_config, bert_config, *inputs, **kwargs):
        super().__init__(bert_config)

        self.args = args

        self.bert = BertModel.from_pretrained(args.lm_path)
        
        for params in self.bert.parameters():
            params.requires_grad = False

        if self.args.pet:
            self.cls = BertPreTrainingHeads(bert_config)

        self.audio_projection = nn.Linear(wav_config.hidden_size, self.args.hidden_size)
        self.text_projection = nn.Linear(bert_config.hidden_size, self.args.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.args.hidden_size)
        # self.dense = nn.Linear(bert_config.hidden_size, self.args.hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(self.args.hidden_size, args.num_labels)
        )

    def forward(self, input_ids, attention_mask,
                token_type_ids, speech_emb):
        
        context_emb = self.bert(input_ids, attention_mask, token_type_ids)[0]
        
        projected_text = self.text_projection(context_emb)
        projected_audio = self.audio_projection(speech_emb)
        
        att_emb = self.dot_attention(projected_text, projected_audio, projected_audio)
        
        if self.args.mm_type == 'concat':
            sequence_output = torch.cat([projected_text, att_emb], dim=1)
        elif self.args.mm_type == 'add':
            sequence_output = att_emb + projected_text
        
        sequence_output = self.LayerNorm(sequence_output)
        # sequence_output = self.dense(sequence_output)
        
        if self.args.opt == 'mean':
            pooled_output = torch.mean(sequence_output, dim=1)
        elif self.args.opt == 'sum':
            pooled_output = torch.sum(sequence_output, dim=1)
        
        class_logit = self.classifier(pooled_output)

        if self.args.pet:
            prediction_scores = self.cls(sequence_output)
            return {
                'hidden_states':sequence_output,
                'pooled_output':pooled_output,
                'prediction_scores':prediction_scores,
                'class_logit':class_logit
            }
        else:
            return {
                'hidden_states':sequence_output,
                'pooled_output':pooled_output,
                'class_logit':class_logit
            }

    def dot_attention(self, q, k, v):
        # q: [bs, bert_l, dim]
        # k=v: [bs, wav_l, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, bert_l, wav_l]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, bert_l, dim]
        return output

class CASECompressingModel(BertPreTrainedModel):
    """
        CASE (compressing)
    """
    def __init__(self, args, wav_config, bert_config):
        super().__init__(bert_config)

        self.args = args
        self.wav_config = wav_config
        self.text_config = bert_config

        self.bert = BertModel.from_pretrained(args.lm_path)
        
        for params in self.bert.parameters():
            params.requires_grad = False

        if self.args.pet:
            self.cls = BertPreTrainingHeads(bert_config)
        
        self.audio_projection = nn.Linear(wav_config.hidden_size, self.args.hidden_size)
        self.text_projection = nn.Linear(bert_config.hidden_size, self.args.hidden_size)

        self.compression_layer = nn.Linear(args.audio_max_len, args.context_max_len)
        self.layer_norm = nn.LayerNorm(self.args.hidden_size)
        # self.dense = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(self.args.hidden_size, args.num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, speech_emb=None):

        text_output = self.bert(input_ids, attention_mask, token_type_ids)[0]
        speech_output = self.padding(speech_emb)

        projected_text = self.text_projection(text_output)
        projected_audio = self.audio_projection(speech_output)

        transposed_audio = projected_audio.transpose(1, 2)
        compressed_audio = self.compression_layer(transposed_audio)
        compressed_audio = compressed_audio.transpose(1, 2)

        if self.args.mm_type == 'concat':
            addition_output = torch.cat([projected_text, compressed_audio], dim=1)
        elif self.args.mm_type == 'add':
            addition_output = projected_text + compressed_audio
            
        addition_output = self.layer_norm(addition_output)
        # addition_output = self.dense(addition_output)
        
        if self.args.opt == 'mean':
            pooled_output = addition_output.mean(dim=1)
        elif self.args.opt == 'sum':
            pooled_output = addition_output.sum(dim=1)
        
        class_logit = self.classifier(pooled_output)
        
        if self.args.pet:
            prediction_scores = self.cls(addition_output)
            
            return {
                "hidden_states": addition_output,
                "pooled_output": pooled_output,
                "prediction_scores":prediction_scores,
                "class_logit": class_logit
            }
        else:
            return {
                "hidden_states": addition_output,
                "pooled_output": pooled_output,
                "class_logit": class_logit
            }

    def padding(self, speech_embedding):

        batch_speech_embedding = torch.Tensor().to(self.args.device)

        for se in speech_embedding:

            se = se.unsqueeze(0)

            sequence_length = se.size()[1]
            if sequence_length >= self.args.audio_max_len:
                se = se[:, :self.args.audio_max_len, :].to(self.args.device)
            else:
                pad = torch.Tensor([[[0]*self.wav_config.hidden_size]*(self.args.audio_max_len-sequence_length)]).to(self.args.device)
                se = torch.cat([se, pad], dim=1)
            
            batch_speech_embedding = torch.cat([batch_speech_embedding, se], dim=0)
        
        return batch_speech_embedding
    
class ConcatModel(BertPreTrainedModel):
    def __init__(self, args, wav_config, bert_config):
        super().__init__(bert_config)

        self.args = args
        self.wav_config = wav_config
        self.text_config = bert_config

        self.bert = BertModel.from_pretrained(args.lm_path)
        
        for params in self.bert.parameters():
            params.requires_grad = False

        self.cls = BertPreTrainingHeads(bert_config)
        
        self.audio_projection = nn.Linear(wav_config.hidden_size, self.args.hidden_size)
        self.text_projection = nn.Linear(bert_config.hidden_size, self.args.hidden_size)

        self.blend_layer = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.layer_norm = nn.LayerNorm(self.args.hidden_size)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(self.args.hidden_size, args.num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, speech_emb=None):

        text_output = self.bert(input_ids, attention_mask, token_type_ids)[0]
        speech_output = self.padding(speech_emb)

        projected_text = self.text_projection(text_output)
        projected_audio = self.audio_projection(speech_output)

        concat_output = torch.cat([projected_text, projected_audio], dim=1)
        
        concat_output = self.blend_layer(concat_output)
        concat_output = self.layer_norm(concat_output)

        pooled_output = concat_output.mean(dim=1)

        class_logit = self.classifier(pooled_output)
        
        return {
            "hidden_states": concat_output,
            "pooled_output": pooled_output,
            "class_logit": class_logit
        }

    def padding(self, speech_embedding):

        batch_speech_embedding = torch.Tensor().to(self.args.device)

        for se in speech_embedding:

            se = se.unsqueeze(0)

            sequence_length = se.size()[1]
            if sequence_length >= self.args.audio_max_len:
                se = se[:, :self.args.audio_max_len, :].to(self.args.device)
            else:
                pad = torch.Tensor([[[0]*self.wav_config.hidden_size]*(self.args.audio_max_len-sequence_length)]).to(self.args.device)
                se = torch.cat([se, pad], dim=1)
            
            batch_speech_embedding = torch.cat([batch_speech_embedding, se], dim=0)
        
        return batch_speech_embedding


class MlpBlock(nn.Module):
    def __init__(self,input_dim,dropout=0.3):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim,input_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(input_dim,input_dim)

    def forward(self, x):
        y = self.fc(x)
        y = self.gelu(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return y


class MixerBlock(nn.Module):
    def __init__(self, input_dim, sequence_length,dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.modal_mixing = MlpBlock(input_dim,dropout)
        self.sequence_mixing = MlpBlock(sequence_length,dropout)

    def transpose(self,x):
        return x.permute(0,2,1)

    def forward(self, x):
        y = self.ln(x)
        y = self.transpose(y)
        y = self.sequence_mixing(y)
        y = self.transpose(y)
        x = x + y
        y = self.ln(y)
        y = self.modal_mixing(y)
        y = y+x
        return y
    
class MultiModalMixer(BertPreTrainedModel):
    def __init__(self, args, wav_config, bert_config):
        super().__init__(bert_config)

        mixer_config = {
            'projection_dim' : 256,
            'output_dim' : 512,
            'num_blocks' : 1,
            'dropout' : 0.1,
        }
        self.args = args
        self.wav_config = wav_config
        self.text_config = bert_config

        self.bert = BertModel.from_pretrained(args.lm_path)
        
        for params in self.bert.parameters():
            params.requires_grad = False

        # self.cls = BertPreTrainingHeads(bert_config)
        
        sequence_length = self.args.context_max_len + self.args.audio_max_len

        self.audio_projection = nn.Linear(wav_config.hidden_size, mixer_config['projection_dim'])
        self.text_projection = nn.Linear(bert_config.hidden_size, mixer_config['projection_dim'])

        self.m_blocks = nn.ModuleList([
            MixerBlock(mixer_config['projection_dim'], sequence_length, mixer_config['dropout']) for i in range(mixer_config['num_blocks'])
        ])

        self.ln = nn.LayerNorm(mixer_config['projection_dim'])

        self.classifier = nn.Sequential(
            nn.Dropout(mixer_config['dropout']),
            nn.Linear(mixer_config['projection_dim'], mixer_config['output_dim']),
            nn.GELU(),
            nn.Dropout(mixer_config['dropout']),
            nn.Linear(mixer_config['output_dim'], args.num_labels)
        )

    def freeze(self):
        self.bert.eval()

    def forward(self, input_ids, attention_mask, token_type_ids, speech_emb=None):

        text_output = self.bert(input_ids, attention_mask, token_type_ids)[0]
        speech_output = self.padding(speech_emb)

        projected_text = self.text_projection(text_output)
        projected_audio = self.audio_projection(speech_output)

        x = torch.cat([projected_text, projected_audio], dim=1)
        
        for block in self.m_blocks:
            x = block(x)

        x = self.ln(x)
        pooled_output = x.mean(dim=1)

        class_logit = self.classifier(pooled_output)
        
        return {
            "hidden_states": x,
            "pooled_output": pooled_output,
            "class_logit": class_logit
        }

    def padding(self, speech_embedding):

        batch_speech_embedding = torch.Tensor().to(self.args.device)

        for se in speech_embedding:

            se = se.unsqueeze(0)

            sequence_length = se.size()[1]
            if sequence_length >= self.args.audio_max_len:
                se = se[:, :self.args.audio_max_len, :].to(self.args.device)
            else:
                pad = torch.Tensor([[[0]*self.wav_config.hidden_size]*(self.args.audio_max_len-sequence_length)]).to(self.args.device)
                se = torch.cat([se, pad], dim=1)
            
            batch_speech_embedding = torch.cat([batch_speech_embedding, se], dim=0)
        
        return batch_speech_embedding

class TextOnlyModel(BertPreTrainedModel):
    def __init__(self, args, bert_config):
        super().__init__(bert_config)

        self.args = args

        self.bert = BertModel.from_pretrained(args.lm_path)
        
        for params in self.bert.parameters():
            params.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(bert_config.hidden_size, self.args.hidden_size),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(self.args.hidden_size, args.num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, speech_emb=None):

        pooled_output = self.bert(input_ids, attention_mask, token_type_ids)[1]

        class_logit = self.classifier(pooled_output)
        
        return {
            "pooled_output": pooled_output,
            "class_logit": class_logit
        }

class SpeechOnlyModel(nn.Module):
    def __init__(self, args, wav_config):
        super().__init__()

        self.args = args

        self.dropout = nn.Dropout()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(wav_config.hidden_size, self.args.hidden_size),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(self.args.hidden_size, args.num_labels)
        )
        
    def forward(self, speech_emb):
        
        pooled_output = torch.mean(speech_emb, dim=1)

        pooled_output = self.dropout(pooled_output)
        class_logit = self.classifier(pooled_output)        

        return {
            'pooled_output':pooled_output,
            'class_logit':class_logit
        }