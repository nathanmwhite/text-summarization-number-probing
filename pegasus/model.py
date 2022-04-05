# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import torch

# TODO: implement Addition and Decoding models

# TODO: update hidden_dim value
class MaxProbingModel(torch.nn.Module):
    def __init__(self, embedding_model):
        super(MaxProbingModel, self).__init__()
        
        self.embedding_model = embedding_model
        
        bilstm_input_dim = self.embedding_model.model.encoder.layer_norm.normalized_shape[0]
        hidden_dim = 5 
        
        # TODO: determine improved implementation of h0 and c0
        self.h0 = torch.randn(2, 5, 5)
        self.c0 = torch.randn(2, 5, 5)
        
        self.bilstm = torch.nn.LSTM(input_size = bilstm_input_dim,
                                    hidden_size = hidden_dim,
                                    num_layers = 1,
                                    bidirectional=True)
        self.linear = torch.nn.Linear(in_features=hidden_dim, 
                                      out_features=1) # 1 is from the orig code
        
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
        # "...a weight matrix and softmax function assign a probability to each index using the model’s hidden state."
        logits = self.linear(hidden_vectors[1][0]).squeeze(-1)
        
        # TODO: review choice to use log_softmax here,
        #  as pytorch's CrossEntropyLoss implicitly applies softmax and log itself
        #  This will especially need to be revisited once code for metrics is written.
        y_pred = F.log_softmax(logits, dim=1)
        
        return y_pred
