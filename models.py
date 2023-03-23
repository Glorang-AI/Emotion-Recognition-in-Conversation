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


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score
    
class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score



class CASEmodel(BertPreTrainedModel):
    """
    Contextual Acoustic Speech Embedding (CASE) model
    """
    def __init__(self, sep_token_id, am_hidden_size, config, *inputs, **kwargs):
        super().__init__(config)
        self.sep_token_id = sep_token_id
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.dense = nn.Linear(am_hidden_size, config.hidden_size)
        self.sentence_embeddings = nn.Embedding(2, config.hidden_size) # Do not have sentence type embedding in klue model(klue/roberta, klue/bert)
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
    def forward(self, input_ids, input_mask,
                    sentence_type_ids, audio_embeddings):
        
        bert_emb = self.bert.embeddings(input_ids)
        bert_emb += self.sentence_embeddings(sentence_type_ids)
        output = self.bert.encoder(bert_emb,
                            return_dict=True)[0] # last hidden states
        
        _, sep_pos = torch.where(input_ids==self.sep_token_id)
        sep_pos = sep_pos[::2] #skip last sep
        txt_wav_att=[]
        pos_idx=0
        
        for batch_n in range(input_ids.shape[0]):
            if type(audio_embeddings) == tuple:
                pos = sep_pos[pos_idx]
                pos_idx+=1
                senten_1 = output[batch_n, :pos, :].unsqueeze(0)
                senten_2 = output[batch_n, pos:, :].unsqueeze(0)
                
                s1_audio, s2_audio = audio_embeddings
                
                s1_audio = self.dense(s1_audio[batch_n]).unsqueeze(0)
                s2_audio = self.dense(s2_audio[batch_n]).unsqueeze(0)
                
                s1_audio_att = self.dot_attention(senten_1, s1_audio, s1_audio).squeeze()
                s2_audio_att = self.dot_attention(senten_2, s2_audio, s2_audio).squeeze()
                
                txt_wav_att.append(torch.cat([s1_audio_att, s2_audio_att], dim=0))
            
            else:
                senten_1 = output[batch_n].unsqueeze(0)
                s1_audio= audio_embeddings
                s1_audio = self.dense(s1_audio).unsqueeze(0)
                
                txt_wav_att.append(self.dot_attention(senten_1, s1_audio, s1_audio).squeeze())
           
        sequence_output = output + torch.stack(txt_wav_att, dim=0)
        
        pooled_output = self.pooler(torch.sum(sequence_output, dim=1))
        
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        return {
            'hidden_states':sequence_output,
            'pooled_output':pooled_output,
            'prediction_scores':prediction_scores,
            'seq_relationship_score':seq_relationship_score
        }

    def dot_attention(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output