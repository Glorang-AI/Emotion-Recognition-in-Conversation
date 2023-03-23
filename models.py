import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN, gelu
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel

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

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias
        
class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    
class CASEmodel(BertPreTrainedModel):
    """
    Contextual Acoustic Speech Embedding (CASE) model
    """
    def __init__(self, bert_pth, wav_config, bert_config, num_labels, *inputs, **kwargs):
        super().__init__(bert_config)

        self.bert = BertModel.from_pretrained(bert_pth)
            
        self.cls = BertPreTrainingHeads(bert_config)
        self.convert_dim = nn.Linear(wav_config.hidden_size, bert_config.hidden_size)
        self.dense = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.LayerNorm = nn.LayerNorm(bert_config.hidden_size, eps=bert_config.layer_norm_eps)
        
        self.pooler = nn.Sequential(
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
            nn.Tanh()
        )
        self.classifier = CLSmodel(num_labels, bert_config)
        
    def forward(self, input_ids, attention_mask,
                token_type_ids, speech_emb=None):
        
        context_emb = self.bert(input_ids, attention_mask, token_type_ids)[0]
        
        if speech_emb != None:
            speech_emb = self.convert_dim(speech_emb)            
            att_emb = self.dot_attention(context_emb, speech_emb, speech_emb)
            sequence_output = context_emb + att_emb
            sequence_output = self.LayerNorm(sequence_output)
        else:
            sequence_output = context_emb
            
        sequence_output = self.dense(sequence_output)
        
        pooled_output = self.pooler(torch.sum(sequence_output, dim=1))
        class_logit = self.classifier(pooled_output)
        # prediction_scores = self.cls(sequence_output)

        return {
            'hidden_states':sequence_output,
            'pooled_output':pooled_output,
            # 'prediction_scores':prediction_scores,
            'class_logit':class_logit
        }

    def dot_attention(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output

class ORGmodel(BertPreTrainedModel):
    """
    Contextual Acoustic Speech Embedding (CASE) model
    """
    def __init__(self, bert_pth, wav_config, bert_config, *inputs, **kwargs):
        super().__init__(bert_config)

        self.bert = BertModel.from_pretrained(bert_pth)
            
        self.cls = BertPreTrainingHeads(bert_config)
        self.convert_dim = nn.Linear(wav_config.hidden_size, bert_config.hidden_size)
        self.dense = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.LayerNorm = nn.LayerNorm(bert_config.hidden_size, eps=bert_config.layer_norm_eps)
        
        self.pooler = nn.Sequential(
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
            nn.Tanh()
        )
        
    def forward(self, input_ids, attention_mask,
                token_type_ids=None):
        sequence_output = self.bert(input_ids, attention_mask, token_type_ids)[0]
        pooled_output = self.pooler(torch.sum(sequence_output, dim=1))
        prediction_scores = self.cls(sequence_output)

        return {
            'hidden_states':sequence_output,
            'pooled_output':pooled_output,
            'prediction_scores':prediction_scores
        }

class CLSmodel(nn.Module):
    def __init__(self, num_labels, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.activation = nn.Tanh()
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, pooler_output):
        
        x=pooler_output
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x

# class NCLSmodel(nn.Module):
#     def __init__(self, num_labels, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.activation = nn.Tanh()
#         self.out_proj = nn.Linear(config.hidden_size, num_labels)
    
#     def forward(self, pooler_output):
#         x = self.dropout(pooler_output)
#         x = self.dense(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
        
#         return x
    
class RoCASEmodel(RobertaPreTrainedModel):
    """
    Contextual Acoustic Speech Embedding (CASE) model
    """
    def __init__(self, bert_pth, wav_config, bert_config, num_labels, *inputs, **kwargs):
        super().__init__(bert_config)

        self.roberta = RobertaModel.from_pretrained(bert_pth)
        
        self.lm_head = RobertaLMHead(bert_config)
        self.convert_dim = nn.Linear(wav_config.hidden_size, bert_config.hidden_size)
        self.dense = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.LayerNorm = nn.LayerNorm(bert_config.hidden_size, eps=bert_config.layer_norm_eps)
        
        self.pooler = nn.Sequential(
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
            nn.Tanh()
        )
        self.classifier = CLSmodel(num_labels, bert_config)
        
    def forward(self, input_ids, attention_mask,
                token_type_ids, speech_emb=None):
        
        context_emb = self.roberta(input_ids, attention_mask, token_type_ids)[0]
        
        if speech_emb != None:
            speech_emb = self.convert_dim(speech_emb)            
            att_emb = self.dot_attention(context_emb, speech_emb, speech_emb)
            sequence_output = context_emb + att_emb
            sequence_output = self.LayerNorm(sequence_output)
        else:
            sequence_output = context_emb
            
        sequence_output = self.dense(sequence_output)
        
        pooled_output = self.pooler(torch.sum(sequence_output, dim=1))
        prediction_scores = self.lm_head(sequence_output)
        class_logit = self.classifier(pooled_output)
        return {
            'hidden_states':sequence_output,
            'pooled_output':pooled_output,
            'prediction_scores':prediction_scores,
            'class_logit':class_logit
        }

    def dot_attention(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output
