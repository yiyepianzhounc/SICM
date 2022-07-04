# -*- coding: utf-8 -*-

"""
implement of SICM
"""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import gelu, gelu_new, ACT2FN
from transformers.configuration_bert import BertConfig


from transformers.modeling_outputs import BaseModelOutputWithPooling

from module.crf import CRF
from module.bilstm import BiLSTM

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from transformers.modeling_bert import BertAttention, BertIntermediate, BertOutput, load_tf_weights_in_bert, BertModel

BertLayerNorm = torch.nn.LayerNorm

from utils.utils import gather_indexes

class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type, boundary embeddings
    """
    def __init__(self, config):  
        super().__init__()
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)  
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)  
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)  
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
    
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, boundary_ids=None, inputs_embeds=None):

        if input_ids is not None:
            input_shape = input_ids.size()  
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)  

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Wg = nn.Linear(config.hidden_size, config.att_hidden_size)
        self.Wh = nn.Linear(config.hidden_size, config.att_hidden_size)
        
        self.att_transform = nn.Linear(config.att_hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.W = nn.Linear(config.hidden_size + config.att_hidden_size, config.hidden_size)
        self.init_weights()

        self.add_final_gate = config.add_final_gate
        if self.add_final_gate:
            self.final_gate = Gate(config)

    def init_weights(self):
        nn.init.xavier_uniform_(self.Wg.weight.data)
        nn.init.xavier_uniform_(self.Wh.weight.data)
        
        nn.init.xavier_uniform_(self.att_transform.weight.data)
        nn.init.xavier_uniform_(self.W.weight.data)

    def global_sentence(self, bert_outputs, attention_mask):  
        mask_ = attention_mask.masked_fill(attention_mask==0, -1e9)
        score = torch.nn.Softmax(-1)(mask_.float())
        return torch.matmul(score.unsqueeze(1), bert_outputs).squeeze(1)

    def forward(self, bert_outputs, attention_mask):
        outputs = self.Wh(bert_outputs)  

        attention_mask_ = attention_mask.squeeze()  
        global_sentence = self.global_sentence(bert_outputs, attention_mask_)  
        global_sentence = self.Wg(global_sentence)  
        global_sentence = global_sentence.unsqueeze(1).expand_as(outputs)  

        mix = torch.nn.Tanh()(outputs + global_sentence)  
        weight = self.att_transform(mix).squeeze()  
        weight.masked_fill_(attention_mask_==0, -1e9)  
        weight_ = torch.softmax(weight, -1)  

        att_sentence = torch.bmm(weight_.unsqueeze(1), outputs).squeeze(1)
        att_sentence = att_sentence.unsqueeze(1).expand(bert_outputs.shape[0], bert_outputs.shape[1], -1)

        if self.add_final_gate:
            sequence_output = self.final_gate(bert_outputs, att_sentence)
        else:
            sequence_output = bert_outputs + att_sentence

        return sequence_output


class FeatureAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Wf = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.Wf.weight.data)

    def forward(self, layer_output, feature_matrix, attention_mask):  
        layer_output = self.Wf(layer_output)
        layer_output = self.act(layer_output)  
        layer_output = layer_output.unsqueeze(2)  
        weight = torch.matmul(layer_output, torch.transpose(feature_matrix, 2, 3))  
        weight = weight.squeeze(2)  
        
        attention_mask = attention_mask.squeeze()  
        attention_mask = attention_mask.unsqueeze(-1).expand_as(weight)  
        weight = weight + (1 - attention_mask.float()) * (-10000.0)
        weight = torch.nn.Softmax(dim=-1)(weight)  
        weight = weight.unsqueeze(-1)
        weighted_features = torch.sum(feature_matrix * weight, dim=2)  

        return weighted_features


class Gate(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.text_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.att_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.gate_linear = nn.Linear(config.hidden_size * 2, 1)
        self.reserved_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.text_linear.weight.data)
        nn.init.xavier_uniform_(self.att_linear.weight.data)
        nn.init.xavier_uniform_(self.gate_linear.weight.data)
        nn.init.xavier_uniform_(self.reserved_linear.weight.data)
        nn.init.xavier_uniform_(self.output_linear.weight.data)

    def forward(self, bert_output, att_sentence):
        
        concat_feat = torch.cat([self.text_linear(bert_output), self.att_linear(att_sentence)], dim=-1)
        
        gate = torch.sigmoid(self.gate_linear(concat_feat))  
        gate = gate.repeat(1, 1, self.hidden_size)  

        reserved_att_sentence = torch.mul(gate, torch.tanh(self.reserved_linear(att_sentence)))  
        
        
        gate_output = self.output_linear(
            torch.cat([bert_output, reserved_att_sentence], dim=-1))  
        return gate_output


class BertLayer(nn.Module):

    def __init__(self, config, has_word_attn=False):  
        super().__init__()

        self.add_phonetic = config.add_phonetic
        self.add_glyph = config.add_glyph
        self.phonetic_dim = config.phonetic_dim
        self.glyph_dim = config.glyph_dim
        
        self.feature_attention = FeatureAttention(config)
        
        self.add_middle_attention = config.add_middle_attention
        if self.add_middle_attention:
            self.middle_attenton = Attention(config)
        self.add_feature_gate = config.add_feature_gate
        if self.add_feature_gate:
            self.feature_gate = Gate(config)

        self.chunk_size_feed_forward = config.chunk_size_feed_forward  
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder  
        self.add_cross_attention = config.add_cross_attention  
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"  
            self.crossattention = BertAttention(config)
        
        self.has_word_attn = has_word_attn
        if self.has_word_attn:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)  
            self.act = nn.Tanh()  

            self.word_transform = nn.Linear(config.word_embed_dim, config.hidden_size)  
            self.word_word_weight = nn.Linear(config.hidden_size, config.hidden_size)   
            attn_W = torch.zeros(config.hidden_size, config.hidden_size)                
            self.attn_W = nn.Parameter(attn_W)                                          
            self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)            
            self.fuse_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  

            self.phonetic_transform = nn.Linear(config.phonetic_dim, config.hidden_size)
            self.glyph_transform = nn.Linear(config.glyph_dim, config.hidden_size)

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,                    
            attention_mask=None,              
            input_word_embeddings=None,       
            input_word_mask=None,             
            head_mask=None,                   
            encoder_hidden_states=None,       
            encoder_attention_mask=None,      
            output_attentions=False,           

            phonetic_features=None,
            glyph_features=None
    ):

        self_attention_outputs = self.attention(
            hidden_states,     
            attention_mask,    
            head_mask,         
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]  
        outputs = self_attention_outputs[1:]
        
        if self.is_decoder and encoder_hidden_states is not None:
            
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]  
            outputs = outputs + cross_attention_outputs[1:]  

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output  
        )  

        if self.has_word_attn:
            assert input_word_mask is not None
            
            word_outputs = self.word_transform(input_word_embeddings)  
            word_outputs = self.act(word_outputs)
            word_outputs = self.word_word_weight(word_outputs)
            word_outputs = self.dropout(word_outputs)

            alpha = torch.matmul(layer_output.unsqueeze(2), self.attn_W)  
            alpha = torch.matmul(alpha, torch.transpose(word_outputs, 2, 3))  
            alpha = alpha.squeeze()  
            alpha = alpha + (1 - input_word_mask.float()) * (-10000.0)  
            alpha = torch.nn.Softmax(dim=-1)(alpha)  
            alpha = alpha.unsqueeze(-1)
            
            weighted_word_embedding = torch.sum(word_outputs * alpha, dim=2)
            
            if self.add_phonetic and not self.add_glyph:
                
                phonetic_features = phonetic_features.float()
                phonetic_features = self.phonetic_transform(phonetic_features)
                phonetic_features = self.act(phonetic_features)
                fused_features = torch.cat((weighted_word_embedding, phonetic_features), dim=-1)
                fused_features = self.fused_transform_phonetic(fused_features)
                layer_output = layer_output + fused_features
            elif not self.add_phonetic and self.add_glyph:
                
                glyph_features = glyph_features.float()
                glyph_features = self.glyph_transform(glyph_features)
                
                layer_output = layer_output + glyph_features

            elif self.add_phonetic and self.add_glyph:
                
                phonetic_features = phonetic_features.float()
                phonetic_features = self.phonetic_transform(phonetic_features)
                phonetic_features = self.act(phonetic_features)
                glyph_features = glyph_features.float()
                glyph_features = self.glyph_transform(glyph_features)
                glyph_features = self.act(glyph_features)

                feature_matrix = torch.cat((weighted_word_embedding.unsqueeze(2), phonetic_features.unsqueeze(2),
                                            glyph_features.unsqueeze(2)), dim=2)
                weighted_features = self.feature_attention(layer_output, feature_matrix, attention_mask)
                layer_output = layer_output + weighted_features
                if self.add_feature_gate:
                    layer_output = self.feature_gate(layer_output, weighted_features)
                
            else:
                layer_output = layer_output + weighted_word_embedding  

            if self.add_middle_attention:
                layer_output = self.middle_attenton(layer_output, attention_mask)

            layer_output = self.dropout(layer_output)
            layer_output = self.fuse_layernorm(layer_output)  

        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):  

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_layers = config.add_layers  

        total_layers = []
        for i in range(config.num_hidden_layers):  
            if i in self.add_layers:
                total_layers.append(BertLayer(config, True))  
            else:
                total_layers.append(BertLayer(config, False))  

        self.layer = nn.ModuleList(total_layers)
    
    def forward(
            self,
            hidden_states,                 
            attention_mask=None,
            input_word_embeddings=None,
            input_word_mask=None,
            head_mask=None,                
            encoder_hidden_states=None,    
            encoder_attention_mask=None,   
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,

            phonetic_features=None,
            glyph_features=None
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None  

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    input_word_embeddings,
                    input_word_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,

                    phonetic_features,
                    glyph_features
                )

            else:
                layer_outputs = layer_module(
                    hidden_states,           
                    attention_mask,          
                    input_word_embeddings,   
                    input_word_mask,         
                    layer_head_mask,         
                    encoder_hidden_states,   
                    encoder_attention_mask,  
                    output_attentions,       

                    phonetic_features,
                    glyph_features
                )
            hidden_states = layer_outputs[0]  
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)  

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        
        
        first_token_tensor = hidden_states[:, 0]  
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPreTrainedModel(PreTrainedModel):

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):  
            
            
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)  
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()  
            module.weight.data.fill_(1.0)  
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class WCBertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super(WCBertModel, self).__init__(config)  

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)         
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()  

    def get_input_embeddings(self):  
        return self.embeddings.word_embeddings 

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)  

    def forward(
        
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        matched_word_embeddings=None,
        matched_word_mask=None,
        boundary_ids=None,             

        phonetic_features=None,
        glyph_features=None,

        position_ids=None,
        head_mask=None,                
        inputs_embeds=None,            
        encoder_hidden_states=None,    
        encoder_attention_mask=None,   
        output_attentions=None,        
        output_hidden_states=None,
        return_dict=None,              
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)  

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            boundary_ids=boundary_ids, inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            input_word_embeddings=matched_word_embeddings,
            input_word_mask=matched_word_mask,
            head_mask=head_mask,                                     
            encoder_hidden_states=encoder_hidden_states,             
            encoder_attention_mask=encoder_extended_attention_mask,  
            output_attentions=output_attentions,                     
            output_hidden_states=output_hidden_states,               
            return_dict=return_dict,                                 

            phonetic_features=phonetic_features,
            glyph_features=glyph_features
        )
        
        sequence_output = encoder_outputs[0]  
        
        pooled_output = self.pooler(sequence_output)  

        if not return_dict:  
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SICM(BertPreTrainedModel):
    def __init__(self, config, pretrained_embeddings, num_labels):  
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]  
        embed_dim = pretrained_embeddings.shape[1]        
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)  
        self.bert = WCBertModel(config)                   
        self.dropout = nn.Dropout(config.HP_dropout)      
        self.num_labels = num_labels                      
        self.hidden2tag = nn.Linear(config.hidden_size, num_labels + 2)  
        self.crf = CRF(num_labels, torch.cuda.is_available())  

        self.add_final_attention = config.add_final_attention
        if self.add_final_attention:
            self.attention = Attention(config)

        self.init_weights()
        
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))  
        print("Load pretrained embedding from file.........")

    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            matched_word_ids=None,
            matched_word_mask=None,
            boundary_ids=None,
            labels=None,

            phonetic_features=None,
            glyph_features=None,

            flag="Train"  
    ):
        matched_word_embeddings = self.word_embeddings(matched_word_ids)  

        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            matched_word_embeddings=matched_word_embeddings,
            matched_word_mask=matched_word_mask,
            boundary_ids=boundary_ids,

            phonetic_features=phonetic_features,
            glyph_features=glyph_features
        )

        sequence_output = outputs[0]

        
        if self.add_final_attention:
            sequence_output = self.attention(sequence_output, attention_mask)

        sequence_output = self.dropout(sequence_output)  
        logits = self.hidden2tag(sequence_output)  

        if flag == 'Train':
            assert labels is not None
            loss = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels)
            _, preds = self.crf._viterbi_decode(logits, attention_mask)
            return (loss, preds)
        elif flag == 'Predict':
            _, preds = self.crf._viterbi_decode(logits, attention_mask)
            return (preds,)


class BertFeatureLSTMCRF(BertPreTrainedModel):

    def __init__(self, config, pretrained_embeddings, num_labels):
        super().__init__(config)

        self.add_phonetic = config.add_phonetic
        self.add_glyph = config.add_glyph
        self.phonetic_dim = config.phonetic_dim
        self.glyph_dim = config.glyph_dim
        self.feature_layernorm = torch.nn.LayerNorm(self.phonetic_dim + self.glyph_dim, eps=config.layer_norm_eps)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.HP_dropout)

        self.act = nn.Tanh()
        self.word_transform = nn.Linear(config.word_embed_dim, config.hidden_size)
        self.word_word_weight = nn.Linear(config.hidden_size, config.hidden_size)
        self.bilstm = BiLSTM(config.hidden_size * 2, config.lstm_size, config.HP_dropout)  

        if self.add_phonetic and self.add_glyph:
            self.feature_bilstm = BiLSTM(config.hidden_size*2+self.phonetic_dim+self.glyph_dim, config.lstm_size, config.HP_dropout)

        self.add_final_attention = config.add_final_attention
        if self.add_final_attention:
            self.attention = Attention(config)

        attn_W = torch.zeros(config.hidden_size, config.hidden_size)
        self.attn_W = nn.Parameter(attn_W)
        self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)

        self.num_labels = num_labels
        self.hidden2tag = nn.Linear(config.lstm_size * 2, num_labels + 2)
        self.crf = CRF(num_labels, torch.cuda.is_available())

        self.init_weights()
        
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        print("Load pretrained embedding from file.........")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            matched_word_ids=None,
            matched_word_mask=None,
            boundary_ids=None,
            labels=None,

            phonetic_features=None,
            glyph_features=None,

            flag="Train"
    ):
        matched_word_embeddings = self.word_embeddings(matched_word_ids)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        matched_word_embeddings = self.word_transform(matched_word_embeddings)
        matched_word_embeddings = self.act(matched_word_embeddings)
        matched_word_embeddings = self.word_word_weight(matched_word_embeddings)
        matched_word_embeddings = self.dropout(matched_word_embeddings)

        alpha = torch.matmul(sequence_output.unsqueeze(2), self.attn_W)  
        alpha = torch.matmul(alpha, torch.transpose(matched_word_embeddings, 2, 3))  
        alpha = alpha.squeeze()  
        alpha = alpha + (1 - matched_word_mask.float()) * (-2 ** 31 + 1)
        alpha = torch.nn.Softmax(dim=-1)(alpha)  
        alpha = alpha.unsqueeze(-1)  
        matched_word_embeddings = torch.sum(matched_word_embeddings * alpha, dim=2)  

        sequence_output = torch.cat((sequence_output, matched_word_embeddings), dim=-1)

        if self.add_glyph and self.add_phonetic:
            phonetic_features = phonetic_features.float()
            phonetic_features = self.act(phonetic_features)
            glyph_features = glyph_features.float()
            glyph_features = self.act(glyph_features)
            features_concat = torch.cat((phonetic_features, glyph_features), dim=-1)
            sequence_output = torch.cat((sequence_output, features_concat), dim=-1)

        sequence_output = self.dropout(sequence_output)

        if self.add_phonetic and self.add_glyph:
            lstm_output = self.feature_bilstm(sequence_output, attention_mask)  
        if not self.add_phonetic and not self.add_glyph:
            lstm_output = self.bilstm(sequence_output, attention_mask)

        logits = self.hidden2tag(lstm_output)

        if flag == 'Train':
            assert labels is not None
            loss = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels)
            _, preds = self.crf._viterbi_decode(logits, attention_mask)
            return (loss, preds)
        elif flag == 'Predict':
            _, preds = self.crf._viterbi_decode(logits, attention_mask)
            return (preds,)


class BertBiLSTMCRF(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.HP_dropout)

        self.act = nn.Tanh()

        self.bilstm = BiLSTM(config.hidden_size, config.lstm_size, config.HP_dropout)  

        self.num_labels = num_labels
        self.hidden2tag = nn.Linear(config.lstm_size * 2, num_labels + 2)
        self.crf = CRF(num_labels, torch.cuda.is_available())

        self.init_weights()

    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            matched_word_ids=None,
            matched_word_mask=None,
            boundary_ids=None,
            labels=None,

            phonetic_features=None,
            glyph_features=None,

            flag="Train"
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        lstm_output = self.bilstm(sequence_output, attention_mask)  
        logits = self.hidden2tag(lstm_output)

        if flag == 'Train':
            assert labels is not None
            loss = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels)
            _, preds = self.crf._viterbi_decode(logits, attention_mask)
            return (loss, preds)
        elif flag == 'Predict':
            _, preds = self.crf._viterbi_decode(logits, attention_mask)
            return (preds,)
