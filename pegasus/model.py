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


def report_phase(message):
    timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    formatted_message = f"{timestamp} | {message}"
    print(formatted_message)
    logging.info(formatted_message)


def freeze_module(module, module_type):
    def freeze_component(component):
        for (name, module_) in component.named_children():
            if name in operational_layer_types:
                for param in module_.parameters():
                    param.requires_grad = False
                report_phase(f'Parameter {name} frozen.')
            else: # numbered layers, expandable_layer_types
                freeze_component(module_)
    
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
        
    freeze_component(module)
    report_phase(f'Parameter freezing successful.')
    
    
def layers_generator(embedding_model):
    for layer in embedding_model.model.encoder.layers:
        yield layer
        
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
    def __init__(self, embedding_model, hidden_dim=5):
        super(MaxProbingModel, self).__init__()
        
        self.embedding_model = embedding_model
        
        encoder = self.embedding_model.model.encoder
        bilstm_input_dim = encoder.layer_norm.normalized_shape[0]
        
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
        self.linear = torch.nn.Linear(in_features=hidden_dim*2, 
                                      out_features=1) 
        
    def forward(self, input_text):
        forward = self.embedding_model.model.forward(**input_text)
        encoder_state = forward.encoder_last_hidden_state
        # use torch.Tensor as input, not numpy, otherwise
        #  error will be thrown related to size and 'int'
        embeddings = encoder_state.detach()[:, :-1]
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
        logits = self.linear(hidden_vectors[0]).squeeze(-1)
        
        # TODO: review choice to use log_softmax here,
        #  as pytorch's CrossEntropyLoss implicitly applies softmax and
        #      log itself
        #  This will especially need to be revisited once code for 
        #      metrics is written.
        y_pred = torch.nn.functional.log_softmax(logits, dim=1)
        
        return y_pred

    
class DecodingModel(torch.nn.Module):
    def __init__(self, embedding_model):
        super(DecodingModel, self).__init__()

        self.embedding_model = embedding_model

        encoder = self.embedding_model.model.encoder
        input_dim = encoder.layer_norm.normalized_shape[0]
        hidden_dim = 50

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
        forward = self.embedding_model.model.forward(**input_text)
        encoder_state = forward.encoder_last_hidden_state

        embeddings = encoder_state.detach()[:, :-1]

        y_pred = self.sequential(embeddings).squeeze(-1)

        return y_pred
    
    
class AdditionModel(torch.nn.Module):
    def __init__(self, embedding_model):
        super(AdditionModel, self).__init__()

        self.embedding_model = embedding_model

        encoder = self.embedding_model.model.encoder
        input_dim = encoder.layer_norm.normalized_shape[0] * 2
        hidden_dim = 50

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
        forward = self.embedding_model.model.forward(**input_text)
        encoder_state = forward.encoder_last_hidden_state

        embeddings = encoder_state.detach()[:, :-1]

        embeddings_concat = torch.flatten(embeddings, start_dim=1)

        y_pred = self.sequential(embeddings_concat).squeeze(-1)

        return y_pred

# TODO: also sentence units model
class UnitsModel(torch.nn.Module):
    def __init__(self, embedding_model, output_dim):
        super(UnitsModel, self).__init__()

        self.embedding_model = embedding_model

        encoder = self.embedding_model.model.encoder
        input_dim = encoder.layer_norm.normalized_shape[0] * 2
        hidden_dim = 50

        # they fail to specify their hidden_dim anywhere
        self.linear_1 = torch.nn.Linear(in_features=input_dim,
                                        out_features=hidden_dim)
        self.linear_2 = torch.nn.Linear(in_features=hidden_dim,
                                        out_features=hidden_dim)
        self.linear_3 = torch.nn.Linear(in_features=hidden_dim,
                                        out_features=output_dim)
        self.sequential = torch.nn.Sequential(self.linear_1,
                                              torch.nn.ReLU(),
                                              self.linear_2,
                                              torch.nn.ReLU(),
                                              self.linear_3)

    # TODO: test dimensionality of the following
    def forward(self, input_text):
        forward = self.embedding_model.model.forward(**input_text)
        encoder_state = forward.encoder_last_hidden_state

        embeddings = encoder_state.detach()[:, :-1]

        embeddings_concat = torch.flatten(embeddings, start_dim=1)

        y_pred = self.sequential(embeddings_concat).squeeze(-1)

        return y_pred

# TODO : determine more appropriate default hidden_dim value    
class ContextUnitsModel(torch.nn.Module):
    def __init__(self, embedding_model, output_dim, hidden_dim=5):
        super(ContextUnitsModel, self).__init__()

        self.embedding_model = embedding_model

        encoder = self.embedding_model.model.encoder
        input_dim = encoder.layer_norm.normalized_shape[0]

        self.bilstm = torch.nn.LSTM(input_size = bilstm_input_dim,
                                    hidden_size = hidden_dim,
                                    num_layers = 1,
                                    bidirectional=True,
                                    batch_first=True)
        
        # hidden_dim*2 because input is from a bidirectional LSTM
        self.linear = torch.nn.Linear(in_features=hidden_dim*2, 
                                      out_features=output_dim)

    # inputs are of sequence [sentence, sep, number]
    def forward(self, input_text):
        forward = self.embedding_model.model.forward(**input_text)
        encoder_state = forward.encoder_last_hidden_state

        embeddings = encoder_state.detach()[:, :-1]

        hidden_vectors = self.bilstm(embeddings)
        logits = self.linear(hidden_vectors[0]).squeeze(-1)
        
        y_pred = torch.nn.functional.log_softmax(logits, dim=1)

        return y_pred

# TODO: test
# TODO: especially test Siamese structure
# TODO: revisit: BiLSTM may be more appropriate
# TODO: this has variable length input by nature;
#    check to ensure this works given in the inputs
class RangeModel(torch.nn.Module):
    def __init__(self, embedding_model, hidden_dim=5):
        super(RangeModel, self).__init__()
        
        self.embedding_model = embedding_model
        
        encoder = self.embedding_model.model.encoder
        input_dim = encoder.layer_norm.normalized_shape[0] * 2
        hidden_dim = 50
        
        self.joined_linear = torch.nn.Linear(in_features=input_dim,
                                             out_features=hidden_dim)
        self.joined_sequential = torch.nn.Sequential(self.joined_linear,
                                                     torch.nn.ReLU)
        
        self.linear_left_1 = torch.nn.Linear(in_features=hidden_dim,
                                             out_features=hidden_dim)
        self.linear_left_2 = torch.nn.Linear(in_features=hidden_dim,
                                             out_features=1)
        self.left_sequential = torch.nn.Sequential(self.linear_left_1,
                                                   torch.nn.ReLU,
                                                   self.linear_left_2)
        
        self.linear_right_1 = torch.nn.Linear(in_features=hidden_dim,
                                              out_features=hidden_dim)
        self.linear_right_2 = torch.nn.Linear(in_features=hidden_dim,
                                              out_features=1)
        self.right_sequential = torch.nn.Sequential(self.linear_right_1,
                                                    torch.nn.ReLU,
                                                    self.linear_right_2)
        
    def forward(self, input_text):
        forward = self.embedding_model.model.forward(**input_text)
        encoder_state = forward.encoder_last_hidden_state
        
        embeddings = encoder_state.detach()[:, :-1]
        
        embeddings_concat = torch.flatten(embeddings, start_dim=1)
        
        joined_out = self.joined_sequential(embeddings_concat)
        y_pred_1 = self.left_sequential(joined_out).squeeze(-1)
        y_pred_2 = self.right_sequential(joined_out).squeeze(-1)
        
        return y_pred_1, y_pred_2
