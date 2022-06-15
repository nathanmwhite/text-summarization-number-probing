# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

from datetime import datetime

import logging

logging.basicConfig(filename='model.log', level=logging.INFO)

import torch

from transformers import PegasusForConditionalGeneration 
from transformers import T5ForConditionalGeneration
from transformers import BartForConditionalGeneration
from transformers import ProphetNetForConditionalGeneration
from transformers import BertModel, BertConfig

from ..s2s_ft.modeling_decoding import BertForSeq2SeqDecoder


def report_phase(message):
    timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    formatted_message = f"{timestamp} | {message}"
    print(formatted_message)
    logging.info(formatted_message)

    
# TODO : run testing to make sure this actually freezes everything intended
def freeze_module(module, module_type):
    def freeze_component(component):
        for (name, module_) in component.named_children():
            if name in operational_layer_types:
                for param in module_.parameters():
                    param.requires_grad = False
#                 report_phase(f'Parameter {name} frozen.')
            else: # numbered layers, expandable_layer_types
                freeze_component(module_)
#                 report_phase(f'Expanding {name}.')
    
    if module_type == 'Pegasus':
        # embed_positions may not be necessary
        operational_layer_types = {'shared',
                                   'embed_tokens',
                                   'embed_positions', 
                                   'k_proj',
                                   'v_proj',
                                   'q_proj',
                                   'out_proj',
                                   'fc1',
                                   'fc2',
                                   'self_attn_layer_norm',
                                   'final_layer_norm',
                                   'layer_norm'}
        expandable_layer_types = {'model',
                                  'encoder',
                                  'decoder',
                                  'layers',
                                  'self_attn'}
    elif module_type in ('T5', 'SSR'): # referred to as T5 below
        operational_layer_types = {'shared',
                                   'embed_tokens',
                                   'q',
                                   'k',
                                   'v', 
                                   'o', 
                                   'relative_attention_bias',
                                   'layer_norm',
                                   'dropout',
                                   'wi',
                                   'wo',
                                   'lm_head',
                                   'final_layer_norm'}
        expandable_layer_types = {'model',
                                  'encoder', 
                                  'block', 
                                  'layer',
                                  'SelfAttention',
                                  'DenseReluDense', 
                                  'decoder', 
                                  'EncDecAttention'}
    elif module_type in ('Bart', 'DistilBart'): # referred to as Bart below
        operational_layer_types = {'shared',
                                   'embed_tokens',
                                   'embed_positions',
                                   'k_proj',
                                   'v_proj',
                                   'q_proj',
                                   'out_proj',
                                   'self_attn_layer_norm',
                                   'fc1',
                                   'fc2',
                                   'final_layer_norm',
                                   'layernorm_embedding',
                                   'encoder_attn_layer_norm',
                                   'lm_head'}
        expandable_layer_types = {'model',
                                  'encoder',
                                  'layers',
                                  'self_attn',
                                  'decoder',
                                  'encoder_attn'}
    elif module_type == 'ProphetNet':
        operational_layer_types = {'word_embeddings',
                                   'position_embeddings',
                                   'embeddings_layer_norm',
                                   'key_proj',
                                   'value_proj',
                                   'query_proj',
                                   'out_proj',
                                   'self_attn_layer_norm',
                                   'intermediate',
                                   'output',
                                   'feed_forward_layer_norm',
                                   'relative_pos_embeddings',
                                   'cross_attn_layer_norm',
                                   'lm_head'}
        expandable_layer_types = {'prophetnet',
                                  'encoder',
                                  'layers',
                                  'self_attn',
                                  'feed_forward',
                                  'decoder',
                                  'cross_attn'}
    elif module_type in ('Bert', 'UniLM'): # UniLM plus intermediate_act_fn from Bert
        operational_layer_types = {'word_embeddings',
                                   'token_type_embeddings',
                                   'position_embeddings',
                                   'LayerNorm',
                                   'dropout',
                                   'query',
                                   'key',
                                   'value',
                                   'dense',
                                   'activation',
                                   'intermediate_act_fn',
                                   'decoder',
                                   'seq_relationship',
                                   'crit_mask_lm',
                                   'crit_next_sent'}
        expandable_layer_types = {'bert',
                                  'embeddings',
                                  'encoder',
                                  'layer',
                                  'attention',
                                  'self',
                                  'output',
                                  'intermediate',
                                  'pooler',
                                  'cls',
                                  'predictions',
                                  'transform'}
        
    freeze_component(module)
#     report_phase(f'Parameter freezing successful.')
    
    
# def layers_generator(embedding_model):
#     for layer in embedding_model.model.encoder.layers:
#         yield layer
        
# the current default is Pegasus
# TODO: restructure to handle other model types
# TODO: for Pegasus, it may be easier just to replace all elements in encoder
#    with just the one layer
# class LayerProbingModel(torch.nn.Module):
#     def __init__(self, embedding_model, hidden_dim=50, layer_idx=0):
#         super(LayerProbingModel, self).__init__()
        
#         self.embed_tokens = embedding_model.embed_tokens
#         self.embed_positions = embedding_model.embed_positions
#         self.embed_scale = embedding_model.embed_scale
#         self.dropout = embedding_model.dropout
#         self.training = embedding_model.training
        
#         self.actionable_layer = embedding_model.encoder.layers[layer_idx]

#         self.linear_1 = torch.nn.Linear(in_features=input_dim,
#                                         out_features=hidden_dim)
#         self.linear_2 = torch.nn.Linear(in_features=hidden_dim,
#                                         out_features=hidden_dim)
#         self.linear_3 = torch.nn.Linear(in_features=hidden_dim,
#                                         out_features=1)
#         self.sequential = torch.nn.Sequential(self.linear_1,
#                                               torch.nn.ReLU(),
#                                               self.linear_2,
#                                               torch.nn.ReLU(),
#                                               self.linear_3)
        
#     def forward(self, input_text):
        
#         # This is based on the original implementation in Transformers
#         if not 'inputs_embeds' in input_text.keys():
#             inputs_embeds = self.embed_tokens(input_text['input_ids']) * self.embed_scale
#         else:
#             inputs_embeds = input_text['inputs_embeds']
            
#         if input_ids is not None:
#             input_shape = input_ids.size()
#             input_ids = input_ids.view(-1, input_shape[-1])
        
#         embed_pos = self.embed_positions(input_shape)

#         hidden_states = inputs_embeds + embed_pos

#         hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            
            
#         forward = self.embedding_model.model.forward(**input_text)
#         encoder_state = forward.encoder_last_hidden_state

#         embeddings = encoder_state.detach()[:, :-1]

#         y_pred = self.sequential(embeddings).squeeze(-1)

#         return y_pred


class MaxProbingModel(torch.nn.Module):
    def __init__(self, embedding_model, padded_seq_len=5, hidden_dim=5):
        super(MaxProbingModel, self).__init__()
        
        self.embedding_model = embedding_model
        
        if type(self.embedding_model) == PegasusForConditionalGeneration:
            self.embedding_type = 'Pegasus'
            encoder = self.embedding_model.model.encoder
            bilstm_input_dim = encoder.layer_norm.normalized_shape[0]
            self.has_start_token = False
        elif type(self.embedding_model) == T5ForConditionalGeneration:
            self.embedding_type = 'T5'
            encoder = self.embedding_model.encoder
            # specific to T5-small; T5-base last is block[11]
            bilstm_input_dim = encoder.block[11].layer[1].DenseReluDense.wo.out_features
            self.has_start_token = False
        elif type(self.embedding_model) == BartForConditionalGeneration:
            self.embedding_type = 'Bart'
            encoder = self.embedding_model.model.encoder
            bilstm_input_dim = encoder.layernorm_embedding.normalized_shape[0]
            self.has_start_token = True
        elif type(self.embedding_model) == ProphetNetForConditionalGeneration:
            self.embedding_type = 'ProphetNet'
            encoder = self.embedding_model.prophetnet.encoder
            bilstm_input_dim = encoder.layers[11].feed_forward_layer_norm.normalized_shape[0]
            self.has_start_token = False
        elif type(self.embedding_model) == BertForSeq2SeqDecoder:
            self.embedding_type = 'UniLM'
            encoder = self.embedding_model.bert.encoder
            bilstm_input_dim = encoder.layer[11].output.dense.out_features
            self.has_start_token = True
        elif type(self.embedding_model) == BertModel:
            self.embedding_type = 'Bert'
            encoder = self.embedding_model.encoder
            bilstm_input_dim = encoder.layer[11].output.dense.out_features
            self.has_start_token = True
        
        # TODO: determine improved implementation of h0 and c0
        #     decision: no need: just use torch's default
        # Wallace et al.'s code indicates that they feed the 
        #     output of the bilstm directly into linear
        # deprecated:
        #self.h0 = torch.randn(2, 5, 5)
        #self.c0 = torch.randn(2, 5, 5)
        
        # batch_first to enable easy passing to linear layer, which 
        #     requires batch_dim first
        self.bilstm = torch.nn.LSTM(input_size = bilstm_input_dim,
                                    hidden_size = hidden_dim,
                                    num_layers = 1,
                                    bidirectional=True,
                                    batch_first=True)
        
        # hidden_dim*2 because input is from a bidirectional LSTM
        # output 1 is from the orig code, and produces exactly one 
        #     output per word
        # this logic does not work with the tokenizer behavior:
        #  the individual numbers are broken up into multiple tokens,
        #  meaning that output 1 per word is impossible
        self.linear = torch.nn.Linear(in_features=hidden_dim*2*padded_seq_len, 
                                      out_features=5)
        
    def forward(self, input_text):
        if self.embedding_type == 'Pegasus':
            forward = self.embedding_model.model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'T5':
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'Bart':
            forward = self.embedding_model.model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'ProphetNet':
            input_text = {k: v for (k, v) in input_text.items() if k != 'token_type_ids'}
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'UniLM':
#             length_ = input_text.input_ids.size(1)
#             input_text['position_ids'] = torch.arange(length_, dtype=torch.long)
            forward = self.embedding_model.bert.embeddings.forward(input_text['input_ids'])
            forward = self.embedding_model.bert.encoder.forward(forward, input_text['attention_mask'])
            encoder_state = forward[-1]
        # TODO: test
        elif self.embedding_type == 'Bert':
            input_text = {k: v for (k, v) in input_text.items() if k != 'decoder_input_ids'}
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.last_hidden_state
        # there may be a problem with padding here
        # use torch.Tensor as input, not numpy, otherwise
        #  error will be thrown related to size and 'int'
        if self.has_start_token:
            start = 1
        else:
            start = 0
        embeddings = encoder_state.detach()[:, start:-1]
        # note:
          # self.h0 needs to be the initial hidden state for each element
          #  in the input sequence; likewise for self.c0
        
        hidden_vectors = self.bilstm(embeddings)
        # .size is a method, not a property (unlike tensorflow shape)
        # hidden_vectors[0] has size [1, 5, 10]
        # From Wallace et al. (2019):
        # "...a weight matrix and softmax function assign a probability
        #     to each index using the model’s hidden state."
        #  this is ambiguous between the embedding model and the probing
        #     model 
        #  as well as between the model output versus the actual hidden 
        #      states of the model
        # likewise, they do not specify h0 or c0 in their code, and do 
        #     not feed any into the bilstm
        embeddings_concat = torch.flatten(hidden_vectors[0], start_dim=1)
        logits = self.linear(embeddings_concat).squeeze(-1)
        
        # TODO: review choice to use log_softmax here,
        #  as pytorch's CrossEntropyLoss implicitly applies softmax and
        #      log itself
        #  This will especially need to be revisited once code for 
        #      metrics is written.
        
        # testing only
        #print('Logits size:', logits.size())
        
        y_pred = torch.nn.functional.log_softmax(logits, dim=1)
        
        return y_pred

    
class DecodingModel(torch.nn.Module):
    def __init__(self, embedding_model, padded_seq_len=1):
        super(DecodingModel, self).__init__()

        self.embedding_model = embedding_model

        if type(self.embedding_model) == PegasusForConditionalGeneration:
            self.embedding_type = 'Pegasus'
            encoder = self.embedding_model.model.encoder
            input_dim = encoder.layer_norm.normalized_shape[0] * padded_seq_len
            self.has_start_token = False
        elif type(self.embedding_model) == T5ForConditionalGeneration:
            self.embedding_type = 'T5'
            encoder = self.embedding_model.encoder
            input_dim = encoder.block[11].layer[1].DenseReluDense.wo.out_features * padded_seq_len
            self.has_start_token = False
        elif type(self.embedding_model) == BartForConditionalGeneration:
            self.embedding_type = 'Bart'
            encoder = self.embedding_model.model.encoder
            input_dim = encoder.layernorm_embedding.normalized_shape[0] * padded_seq_len
            self.has_start_token = True
        elif type(self.embedding_model) == ProphetNetForConditionalGeneration:
            self.embedding_type = 'ProphetNet'
            encoder = self.embedding_model.prophetnet.encoder
            input_dim = encoder.layers[11].feed_forward_layer_norm.normalized_shape[0] * padded_seq_len
            self.has_start_token = False
        elif type(self.embedding_model) == BertForSeq2SeqDecoder:
            self.embedding_type = 'UniLM'
            encoder = self.embedding_model.bert.encoder
            input_dim = encoder.layer[11].output.dense.out_features * padded_seq_len
            self.has_start_token = True
        elif type(self.embedding_model) == BertModel:
            self.embedding_type = 'Bert'
            encoder = self.embedding_model.encoder
            input_dim = encoder.layer[11].output.dense.out_features
            self.has_start_token = True
        
        #hidden_dim = 50 # tests pre-May 30 were hidden_dim=50
        hidden_dim = 100

        # their description suggests ReLU at every layer,
        #  though their implementation only has it for first two,
        #  with no activation for the third
        # they fail to specify their hidden_dim anywhere
        self.linear_1 = torch.nn.Linear(in_features=input_dim,
                                        out_features=hidden_dim)
        self.linear_2 = torch.nn.Linear(in_features=hidden_dim,
                                        out_features=hidden_dim)
        self.linear_3 = torch.nn.Linear(in_features=hidden_dim,
                                        out_features=1)
        self.sequential = torch.nn.Sequential(self.linear_1,
                                              torch.nn.ReLU(),
                                              self.linear_2,
                                              torch.nn.ReLU(),
                                              self.linear_3)

    def forward(self, input_text):
        if self.embedding_type == 'Pegasus':
            forward = self.embedding_model.model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'T5':
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'Bart':
            forward = self.embedding_model.model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'ProphetNet':
            input_text = {k: v for (k, v) in input_text.items() if k != 'token_type_ids'}
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'UniLM':
#             length_ = input_text.input_ids.size(1)
#             input_text['position_ids'] = torch.arange(length_, dtype=torch.long)
            forward = self.embedding_model.bert.embeddings.forward(input_text['input_ids'])
            forward = self.embedding_model.bert.encoder.forward(forward, input_text['attention_mask'])
            encoder_state = forward[-1]
        elif self.embedding_type == 'Bert':
            input_text = {k: v for (k, v) in input_text.items() if k != 'decoder_input_ids'}
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.last_hidden_state

        # slice off start and end tokens
        if self.has_start_token:
            start = 1
        else:
            start = 0
        embeddings = encoder_state.detach()[:, start:-1]
        
        embeddings_concat = torch.flatten(embeddings, start_dim=1)

        y_pred = self.sequential(embeddings_concat)
        
        y_pred = y_pred.squeeze(-1)
        
        # testing only
#         print('Embeddings_out dims:', embeddings.size())
#         print('Y_pred dims:', y_pred.size())

        return y_pred
    
    
class AdditionModel(torch.nn.Module):
    def __init__(self, embedding_model, padded_seq_len=2):
        super(AdditionModel, self).__init__()

        self.embedding_model = embedding_model

        # TODO: test that T5 input type actually works here
        if type(self.embedding_model) == PegasusForConditionalGeneration:
            self.embedding_type = 'Pegasus'
            encoder = self.embedding_model.model.encoder
            input_dim = encoder.layer_norm.normalized_shape[0] * padded_seq_len
            self.has_start_token = False
        elif type(self.embedding_model) == T5ForConditionalGeneration:
            self.embedding_type = 'T5'
            encoder = self.embedding_model.encoder
            input_dim = encoder.block[11].layer[1].DenseReluDense.wo.out_features * padded_seq_len
            self.has_start_token = False
        elif type(self.embedding_model) == BartForConditionalGeneration:
            self.embedding_type = 'Bart'
            encoder = self.embedding_model.model.encoder
            input_dim = encoder.layernorm_embedding.normalized_shape[0] * padded_seq_len
            self.has_start_token = True
        elif type(self.embedding_model) == ProphetNetForConditionalGeneration:
            self.embedding_type = 'ProphetNet'
            encoder = self.embedding_model.prophetnet.encoder
            input_dim = encoder.layers[11].feed_forward_layer_norm.normalized_shape[0] * padded_seq_len
            self.has_start_token = False
        elif type(self.embedding_model) == BertForSeq2SeqDecoder:
            self.embedding_type = 'UniLM'
            encoder = self.embedding_model.bert.encoder
            input_dim = encoder.layer[11].output.dense.out_features * padded_seq_len
            self.has_start_token = True
        elif type(self.embedding_model) == BertModel:
            self.embedding_type = 'Bert'
            encoder = self.embedding_model.encoder
            input_dim = encoder.layer[11].output.dense.out_features
            self.has_start_token = True
        
        #hidden_dim = 50 # pre-May 30
        hidden_dim = 100

        # their description suggests ReLU at every layer,
        #  though their implementation only has it for first two,
        #  with no activation for the third
        # they fail to specify their hidden_dim anywhere
        self.linear_1 = torch.nn.Linear(in_features=input_dim,
                                        out_features=hidden_dim)
        self.linear_2 = torch.nn.Linear(in_features=hidden_dim,
                                        out_features=hidden_dim)
        self.linear_3 = torch.nn.Linear(in_features=hidden_dim,
                                        out_features=1)
        self.sequential = torch.nn.Sequential(self.linear_1,
                                              torch.nn.ReLU(),
                                              self.linear_2,
                                              torch.nn.ReLU(),
                                              self.linear_3)

    # TODO: test dimensionality of the following
    def forward(self, input_text):
        if self.embedding_type == 'Pegasus':
            forward = self.embedding_model.model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'T5':
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'Bart':
            forward = self.embedding_model.model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'ProphetNet':
            input_text = {k: v for (k, v) in input_text.items() if k != 'token_type_ids'}
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'UniLM':
#             length_ = input_text.input_ids.size(1)
#             input_text['position_ids'] = torch.arange(length_, dtype=torch.long)
            forward = self.embedding_model.bert.embeddings.forward(input_text['input_ids'])
            forward = self.embedding_model.bert.encoder.forward(forward, input_text['attention_mask'])
            encoder_state = forward[-1]
        elif self.embedding_type == 'Bert':
            input_text = {k: v for (k, v) in input_text.items() if k != 'decoder_input_ids'}
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.last_hidden_state
        
        if self.has_start_token:
            start = 1
        else:
            start = 0
        embeddings = encoder_state.detach()[:, start:-1]

        embeddings_concat = torch.flatten(embeddings, start_dim=1)

        y_pred = self.sequential(embeddings_concat).squeeze(-1)

        return y_pred

# TODO: Does UnitsModel need a BiLSTM? My original implementation
#  used an MLP, but it lacked a classifier at the last step,
#  which should have been present.
class UnitsModel(torch.nn.Module):
    def __init__(self, embedding_model, output_dim, padded_seq_len=2, hidden_dim=5):
        super(UnitsModel, self).__init__()

        self.embedding_model = embedding_model
        
        if type(self.embedding_model) == PegasusForConditionalGeneration:
            self.embedding_type = 'Pegasus'
            encoder = self.embedding_model.model.encoder
            bilstm_input_dim = encoder.layer_norm.normalized_shape[0]
            self.has_start_token = False
        elif type(self.embedding_model) == T5ForConditionalGeneration:
            self.embedding_type = 'T5'
            encoder = self.embedding_model.encoder
            bilstm_input_dim = encoder.block[11].layer[1].DenseReluDense.wo.out_features
            self.has_start_token = False
        elif type(self.embedding_model) == BartForConditionalGeneration:
            self.embedding_type = 'Bart'
            encoder = self.embedding_model.model.encoder
            bilstm_input_dim = encoder.layernorm_embedding.normalized_shape[0]
            self.has_start_token = True
        elif type(self.embedding_model) == ProphetNetForConditionalGeneration:
            self.embedding_type = 'ProphetNet'
            encoder = self.embedding_model.prophetnet.encoder
            bilstm_input_dim = encoder.layers[11].feed_forward_layer_norm.normalized_shape[0]
            self.has_start_token = False
        elif type(self.embedding_model) == BertForSeq2SeqDecoder:
            self.embedding_type = 'UniLM'
            encoder = self.embedding_model.bert.encoder
            bilstm_input_dim = encoder.layer[11].output.dense.out_features
            self.has_start_token = True
        elif type(self.embedding_model) == BertModel:
            self.embedding_type = 'Bert'
            encoder = self.embedding_model.encoder
            bilstm_input_dim = encoder.layer[11].output.dense.out_features
            self.has_start_token = True
        
        # they fail to specify their hidden_dim anywhere
#         self.linear_1 = torch.nn.Linear(in_features=input_dim,
#                                         out_features=hidden_dim)
#         self.linear_2 = torch.nn.Linear(in_features=hidden_dim,
#                                         out_features=hidden_dim)
#         self.linear_3 = torch.nn.Linear(in_features=hidden_dim,
#                                         out_features=output_dim)
#         self.sequential = torch.nn.Sequential(self.linear_1,
#                                               torch.nn.ReLU(),
#                                               self.linear_2,
#                                               torch.nn.ReLU(),
#                                               self.linear_3)

        self.bilstm = torch.nn.LSTM(input_size = bilstm_input_dim,
                                    hidden_size = hidden_dim,
                                    num_layers = 1,
                                    bidirectional=True,
                                    batch_first=True)
        
        # hidden_dim*2 because input is from a bidirectional LSTM
        self.linear = torch.nn.Linear(in_features=hidden_dim*2*padded_seq_len, 
                                      out_features=output_dim)

    # TODO: test dimensionality of the following
    def forward(self, input_text):
        if self.embedding_type == 'Pegasus':
            forward = self.embedding_model.model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'T5':
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'Bart':
            forward = self.embedding_model.model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'ProphetNet':
            input_text = {k: v for (k, v) in input_text.items() if k != 'token_type_ids'}
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'UniLM':
#             length_ = input_text.input_ids.size(1)
#             input_text['position_ids'] = torch.arange(length_, dtype=torch.long)
            forward = self.embedding_model.bert.embeddings.forward(input_text['input_ids'])
            forward = self.embedding_model.bert.encoder.forward(forward, input_text['attention_mask'])
            encoder_state = forward[-1]
        elif self.embedding_type == 'Bert':
            input_text = {k: v for (k, v) in input_text.items() if k != 'decoder_input_ids'}
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.last_hidden_state

        if self.has_start_token:
            start = 1
        else:
            start = 0
        embeddings = encoder_state.detach()[:, start:-1]

#         y_pred = self.sequential(embeddings_concat).squeeze(-1)

        hidden_vectors = self.bilstm(embeddings)
        embeddings_concat = torch.flatten(hidden_vectors[0], start_dim=1)
        logits = self.linear(embeddings_concat).squeeze(-1)
        
        y_pred = torch.nn.functional.log_softmax(logits, dim=1)

        return y_pred
    

# TODO : determine more appropriate default hidden_dim value    
class ContextUnitsModel(torch.nn.Module):
    def __init__(self, embedding_model, output_dim, padded_seq_len=1, hidden_dim=5):
        super(ContextUnitsModel, self).__init__()

        self.embedding_model = embedding_model
        
        if type(self.embedding_model) == PegasusForConditionalGeneration:
            self.embedding_type = 'Pegasus'
            encoder = self.embedding_model.model.encoder
            bilstm_input_dim = encoder.layer_norm.normalized_shape[0]
            self.has_start_token = False
        elif type(self.embedding_model) == T5ForConditionalGeneration:
            self.embedding_type = 'T5'
            encoder = self.embedding_model.encoder
            bilstm_input_dim = encoder.block[11].layer[1].DenseReluDense.wo.out_features
            self.has_start_token = False
        elif type(self.embedding_model) == BartForConditionalGeneration:
            self.embedding_type = 'Bart'
            encoder = self.embedding_model.model.encoder
            bilstm_input_dim = encoder.layernorm_embedding.normalized_shape[0]
            self.has_start_token = True
        elif type(self.embedding_model) == ProphetNetForConditionalGeneration:
            self.embedding_type = 'ProphetNet'
            encoder = self.embedding_model.prophetnet.encoder
            bilstm_input_dim = encoder.layers[11].feed_forward_layer_norm.normalized_shape[0]
            self.has_start_token = False
        elif type(self.embedding_model) == BertForSeq2SeqDecoder:
            self.embedding_type = 'UniLM'
            encoder = self.embedding_model.bert.encoder
            bilstm_input_dim = encoder.layer[11].output.dense.out_features
            self.has_start_token = True
        elif type(self.embedding_model) == BertModel:
            self.embedding_type = 'Bert'
            encoder = self.embedding_model.encoder
            bilstm_input_dim = encoder.layer[11].output.dense.out_features
            self.has_start_token = True

        self.bilstm = torch.nn.LSTM(input_size = bilstm_input_dim,
                                    hidden_size = hidden_dim,
                                    num_layers = 1,
                                    bidirectional=True,
                                    batch_first=True)
        
        # hidden_dim*2 because input is from a bidirectional LSTM
        self.linear = torch.nn.Linear(in_features=hidden_dim*2*padded_seq_len, 
                                      out_features=output_dim)

    # inputs are of sequence [sentence, sep, number]
    def forward(self, input_text):
        if self.embedding_type == 'Pegasus':
            forward = self.embedding_model.model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'T5':
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'Bart':
            forward = self.embedding_model.model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'ProphetNet':
            input_text = {k: v for (k, v) in input_text.items() if k != 'token_type_ids'}
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'UniLM':
#             length_ = input_text.input_ids.size(1)
#             input_text['position_ids'] = torch.arange(length_, dtype=torch.long)
            forward = self.embedding_model.bert.embeddings.forward(input_text['input_ids'])
            forward = self.embedding_model.bert.encoder.forward(forward, input_text['attention_mask'])
            encoder_state = forward[-1]
        elif self.embedding_type == 'Bert':
            input_text = {k: v for (k, v) in input_text.items() if k != 'decoder_input_ids'}
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.last_hidden_state
            
        if self.has_start_token:
            start = 1
        else:
            start = 0
        embeddings = encoder_state.detach()[:, start:-1]

        hidden_vectors = self.bilstm(embeddings)
        embeddings_concat = torch.flatten(hidden_vectors[0], start_dim=1)
        logits = self.linear(embeddings_concat).squeeze(-1)
        
        y_pred = torch.nn.functional.log_softmax(logits, dim=1)

        return y_pred

    
# TODO: revisit: BiLSTM may be more appropriate
class RangeModel(torch.nn.Module):
    def __init__(self, embedding_model, padded_seq_len=2, hidden_dim=5):
        super(RangeModel, self).__init__()
        
        self.embedding_model = embedding_model
        
        if type(self.embedding_model) == PegasusForConditionalGeneration:
            self.embedding_type = 'Pegasus'
            encoder = self.embedding_model.model.encoder
            input_dim = encoder.layer_norm.normalized_shape[0] * padded_seq_len
            self.has_start_token = False
        elif type(self.embedding_model) == T5ForConditionalGeneration:
            self.embedding_type = 'T5'
            encoder = self.embedding_model.encoder
            input_dim = encoder.block[11].layer[1].DenseReluDense.wo.out_features * padded_seq_len
            self.has_start_token = False
        elif type(self.embedding_model) == BartForConditionalGeneration:
            self.embedding_type = 'Bart'
            encoder = self.embedding_model.model.encoder
            input_dim = encoder.layernorm_embedding.normalized_shape[0] * padded_seq_len
            self.has_start_token = True
        elif type(self.embedding_model) == ProphetNetForConditionalGeneration:
            self.embedding_type = 'ProphetNet'
            encoder = self.embedding_model.prophetnet.encoder
            input_dim = encoder.layers[11].feed_forward_layer_norm.normalized_shape[0] * padded_seq_len
            self.has_start_token = False
        elif type(self.embedding_model) == BertForSeq2SeqDecoder:
            self.embedding_type = 'UniLM'
            encoder = self.embedding_model.bert.encoder
            input_dim = encoder.layer[11].output.dense.out_features * padded_seq_len
            self.has_start_token = True
        elif type(self.embedding_model) == BertModel:
            self.embedding_type = 'Bert'
            encoder = self.embedding_model.encoder
            input_dim = encoder.layer[11].output.dense.out_features
            self.has_start_token = True

        hidden_dim = 50
        
        self.joined_linear = torch.nn.Linear(in_features=input_dim,
                                             out_features=hidden_dim)
        self.joined_sequential = torch.nn.Sequential(self.joined_linear,
                                                     torch.nn.ReLU())
        
        self.linear_left_1 = torch.nn.Linear(in_features=hidden_dim,
                                             out_features=hidden_dim)
        self.linear_left_2 = torch.nn.Linear(in_features=hidden_dim,
                                             out_features=1)
        self.left_sequential = torch.nn.Sequential(self.linear_left_1,
                                                   torch.nn.ReLU(),
                                                   self.linear_left_2)
        
        self.linear_right_1 = torch.nn.Linear(in_features=hidden_dim,
                                              out_features=hidden_dim)
        self.linear_right_2 = torch.nn.Linear(in_features=hidden_dim,
                                              out_features=1)
        self.right_sequential = torch.nn.Sequential(self.linear_right_1,
                                                    torch.nn.ReLU(),
                                                    self.linear_right_2)
        
    def forward(self, input_text):
        if self.embedding_type == 'Pegasus':
            forward = self.embedding_model.model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'T5':
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'Bart':
            forward = self.embedding_model.model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'ProphetNet':
            input_text = {k: v for (k, v) in input_text.items() if k != 'token_type_ids'}
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.encoder_last_hidden_state
        elif self.embedding_type == 'UniLM':
#             length_ = input_text.input_ids.size(1)
#             input_text['position_ids'] = torch.arange(length_, dtype=torch.long)
            forward = self.embedding_model.bert.embeddings.forward(input_text['input_ids'])
            forward = self.embedding_model.bert.encoder.forward(forward, input_text['attention_mask'])
            encoder_state = forward[-1]
        elif self.embedding_type == 'Bert':
            input_text = {k: v for (k, v) in input_text.items() if k != 'decoder_input_ids'}
            forward = self.embedding_model.forward(**input_text)
            encoder_state = forward.last_hidden_state
        
        # slice off start and end tokens
        if self.has_start_token:
            start = 1
        else:
            start = 0
        embeddings = encoder_state.detach()[:, start:-1]
        
        embeddings_concat = torch.flatten(embeddings, start_dim=1)
        
        joined_out = self.joined_sequential(embeddings_concat)
        y_pred_1 = self.left_sequential(joined_out).squeeze(-1)
        y_pred_2 = self.right_sequential(joined_out).squeeze(-1)
        
        return y_pred_1, y_pred_2
