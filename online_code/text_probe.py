# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import argparse

from datetime import datetime

import logging

import math
import os

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

# TODO: update relative imports -- done
from ..pegasus.generate_data import generate_data
from ..pegasus.model import AdditionModel, report_phase, freeze_module
from ..pegasus.util import check_arguments, get_model_name, get_tokenizer, get_embedding_model
from ..pegasus.early_stopping import Early_Stopping

from .online_code import OnlineCode

# assumption: train on portion of data for x epochs, then introduce next group
#  under this assumption, do nothing here
#  online code computations should be done in main loop, with training split into sections
#  with sections of data
def train_epoch(idx, training_data_loader, model, loss_function, optimizer, clip_norm):
    batch_loss = 0.0
    continuing_loss = 0.0
    total_loss = 0.0
    
    for i, data_batch in enumerate(training_data_loader):
        inputs, labels = data_batch
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        # testing only
        #print('Outputs size:', outputs.size())
        #print('Labels size:', labels.size())
        
        loss = loss_function(outputs, labels)
        
        loss.backward()
        
        clip_grad_norm_(filter(lambda x: x.requires_grad, model.parameters()), clip_norm)
                
        optimizer.step()
        
        continuing_loss += loss.item()
        total_loss += loss.item()
        
        if i % 250 == 249:
            batch_loss = continuing_loss / 250
            n = i + 1
            loss_message = f"-- Batch {n} loss: {batch_loss}"
            print(loss_message)
            logging.info(loss_message)
            continuing_loss = 0.0
            
    return batch_loss, continuing_loss, total_loss
  

# This will likely be obsolete when finished
def evaluate(model, metric, eval_dataloader):
    model.eval()
    
    total_loss = 0.0
    
    for i, data_point in enumerate(eval_dataloader):
        inputs, labels = data_point
        
        output = model(inputs)
        
        metric.update_with_results(output, labels)
        
        codelength = get_prequential_codelength()
        
    return codelength
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: update and structure for all tasks, not just addition
    # TODO: add task as argparse argument
    parser.add_argument('--task', type=str, default='Addition')
    parser.add_argument('--embedding_model', type=str, default='Pegasus')
    parser.add_argument('--training_examples', type=int, default=1000)
    parser.add_argument('--test_examples', type=int, default=100)
    parser.add_argument('--sample_min_int', type=int, default=0)
    parser.add_argument('--sample_max_int', type=int, default=99)
    parser.add_argument('--sample_min_float', type=int, default=0)
    parser.add_argument('--sample_max_float', type=int, default=99)
    parser.add_argument('--float', type=bool, default=False)
    parser.add_argument('--use_words', type=bool, default=False)
    parser.add_argument('--num_partitions', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--freeze_embedder', type=bool, default=False)
    parser.add_argument('--log_filename', type=str, default='addition.log')
    parser.add_argument('--trial_number', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--early_stopping', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--clip_norm', type=int, default=5)
    parser.add_argument('--trained', action='store_true')
    parser.add_argument('--untrained', dest='trained', action='store_false')
    parser.set_defaults(trained=True)
    args = parser.parse_args()
    
    check_arguments(args)
    
    logging.basicConfig(filename=args.log_filename, level=logging.INFO)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = get_model_name(args.embedding_model)
    
    tokenizer = get_tokenizer(model_name)

    if args.float:
        sample_min = args.sample_min_float
        sample_max = args.sample_max_float
    else:
        sample_min = args.sample_min_int
        sample_max = args.sample_max_int
    
    # TODO: revisit whether a test component is even relevant
    n_training_examples = args.training_examples
    n_test_examples = args.test_examples
    
    n_partitions = args.num_partitions
    partition_size = math.floor(n_training_examples / n_partitions)
    
    if n_training_examples % n_partitions != 0:
        n_last_portion = partition_size + (n_training_examples % n_partitions)
    
#     phase_message = 'Begin generating dataset.'
#     report_phase(phase_message)
    
    # TODO: substitute appropriate task as arg from above -- done
    training_dataset, test_dataset = generate_data(
        tokenizer, device, sample_min, sample_max,
        n_training_examples, n_test_examples, args.task,
        use_word_format=args.use_words, float_=args.float)
    
    if args.embedding_model in ('Pegasus', 'Pegasus-CDM', 'T5', 'T5-CDM', 'SSR', 'ProphetNet', 'ProphetNet-CDM'):
        start_token_length = 0
    elif args.embedding_model in ('Bert', 'Bart', 'Bart-L', 'Bart-XSum', 'Bart-CDM', 'DistilBart', 'DistilBart-CDM', 'UniLM', 'Random'):
        start_token_length = 1
#     else:
#         raise ValueError('Error: --embedding_model must be a valid model type.')
    
    padded_seq_len = training_dataset[0][0]['input_ids'].size()[-1] - 1 \
        - start_token_length
    
    if args.embedding_model == 'UniLM':
        training_batch_size = 1
    else:
        training_batch_size = args.batch_size
    
    # temporary testing purposes
#     print(training_batch_size)
    
    # TODO: split DataLoader into n dataloaders, -- done
    #  one for each chunk in the online code calculation
    #  for length of each, use partition_size and n_last_portion
    training_dataloaders = []
    for s in range(0, n_partitions - 1): # up to non-final
        phase_message = 'Processing partition size from {a} to {b}'.format(a=partition_size*s, b=partition_size*(s+1))
        report_phase(phase_message)
        training_dataloader = DataLoader(training_dataset[partition_size*s:partition_size*(s+1)], 
                                         batch_size=training_batch_size, 
                                         shuffle=True)
        training_dataloaders.append(training_dataloader)
    phase_message = 'Processing partition size from {a} to {b}'.format(a=partition_size*(n_partitions-1), b=training_dataset[2].size(dim=0))
    report_phase(phase_message)
    training_dataloader = DataLoader(training_dataset[partition_size*(n_partitions-1):],
                                     batch_size=training_batch_size,
                                     shuffle=True)
    training_dataloaders.append(training_dataloader)
    #test_dataloader = DataLoader(test_dataset, 
    #                             batch_size=1, 
    #                             shuffle=True)
    
#     phase_message = 'Completed generating dataset.'
#     report_phase(phase_message)
    
    # TODO: continue checking for changes from here
    embedding_model = get_embedding_model(model_name, args.trained)
        
    if args.freeze_embedder:
        freeze_module(embedding_model, args.embedding_model)
    embedding_model = embedding_model.to(device)
    
    am = AdditionModel(embedding_model, padded_seq_len).to(device)

#     phase_message = 'Model set up.'
#     report_phase(phase_message)
    
    loss_fn = torch.nn.MSELoss()
    
    # likely obsolete:
    # hyperparameters per Wallace et al. (2019) code
    # TODO: learning rate is too big after epoch 35 or so
    # need to implement a LR scheduler
    if args.freeze_embedder:
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, am.parameters()),
                                    lr=args.lr,
                                    momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(am.parameters(), lr=args.lr, momentum=args.momentum)
    
    EPOCHS = args.epochs
    
#     phase_message = 'Begin training.'
#     report_phase(phase_message)

    early_stopping = Early_Stopping(min_delta=0.0, patience=args.patience)
    
    epoch_number = 0
    online_code = OnlineCode(partition_size, n_partitions)
    
    # TODO: epoch loop needs to be reorganized into n cycles, one for each segment of the full data
    #  then codelength and compression need to be called in eval cycle
    for loader in training_dataloaders:
        for epoch in range(EPOCHS):
    #         epoch_message = 'Begin epoch {n}'.format(n=epoch_number + 1)
    #         report_phase(epoch_message)

            # Make sure gradient tracking is on, and do a pass over the data
            am.train(True)
            avg_loss, continuing_loss, total_loss = train_epoch(
                epoch_number, loader, am, loss_fn, optimizer, args.clip_norm
            )

    #         phase_message = f"End of epoch average batch loss: {avg_loss}"
    #         report_phase(phase_message)
    #         phase_message = f"End of epoch last loss: {continuing_loss}"
    #         report_phase(phase_message)
    #         phase_message = f"Epoch total loss: {total_loss}"
    #         report_phase(phase_message)

            if args.early_stopping:
                early_stopping(total_loss)
                if early_stopping.early_stopping == True:
                    message = f'Early stopping of training at epoch {epoch_number}.'
                    report_phase(message)
                    break


            epoch_number += 1
        
    # temporary: save last version of model
    # TODO: reimplement to save best version -- obsolete
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     model_path = f"model_{timestamp}_{epoch_number}"
#     torch.save(am.state_dict(), model_path)
        
    # testing and metrics
#     message = 'Training finished.'
#     report_phase(message)
#     message = 'Begin evaluation.'
#     report_phase(message)
    # TODO: determine whether a separate evaluate function on a test dataset is even necessary -- done
        am.eval()
        with torch.no_grad():
            codelength = evaluate(am, online_code, test_dataloader)
            
    codelength = get_prequential_codelength()
    compression = get_compression(n_training_examples)
    
    # TODO: redo print of results based on codelength and compression as metrics -- done
    hyperparam_set = (f'{args.task} trial',
                      args.embedding_model,
                      args.training_examples,
                      args.lr,
                      args.momentum,
                      args.epochs,
                      args.trial_number)
                      
    message = f"Model hyperparameters: " + ' | '.join(str(w) for w in hyperparam_set)
    report_phase(message)
    message = f"Test Codelength: {codelength}"
    report_phase(message)
    message = f"Test Compression: {compression}"
    report_phase(message)
