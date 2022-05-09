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

import torch
from torch.utils.data import DataLoader

from torchmetrics import Accuracy

from .generate_data import generate_data
from .model import UnitsModel, ContextUnitsModel, report_phase, freeze_module
from .util import check_arguments, get_model_name, get_tokenizer, get_embedding_model


def train_epoch(idx, training_data_loader, model, loss_function, optimizer, num_classes):
    batch_loss = 0.0
    continuing_loss = 0.0
    total_loss = 0.0
    
    accuracy = Accuracy(num_classes=num_classes)
    
    for i, data_batch in enumerate(training_data_loader):
        inputs, labels = data_batch
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        # testing only
        #print('Outputs size:', outputs.size())
        #print('Labels size:', labels.size())
        
        loss = loss_function(outputs, labels)
        
        loss.backward()
        
        label_int_tensor = torch.argmax(labels, axis=-1)
        
        # testing only
        #print(label_int_tensor.device)
        #print(outputs.device)
        
        # torchmetrics implementation requires transfer to CPU
        labels_cpu = label_int_tensor.to("cpu")
        outputs_cpu = outputs.to("cpu")
        
        batch_accuracy = accuracy(outputs_cpu, labels_cpu)
                
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
            
    return batch_loss, continuing_loss, accuracy.compute()


def evaluate(model, eval_dataloader):
    model.eval()
    accuracy = Accuracy()
    
    for i, data_point in enumerate(eval_dataloader):
        inputs, labels = data_point
        
        output = model(inputs)
        
        label_int_tensor = torch.argmax(labels, axis=-1)
        
        # torchmetrics implementation requires transfer to CPU
        labels_cpu = label_int_tensor.to("cpu")
        outputs_cpu = output.to("cpu")
        
        _ = accuracy(outputs_cpu, labels_cpu)
        
    return accuracy.compute()
  
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_model', type=str, default='Pegasus')
    parser.add_argument('--training_examples', type=int, default=1000)
    parser.add_argument('--test_examples', type=int, default=100)
    parser.add_argument('--sample_min_int', type=int, default=0)
    parser.add_argument('--sample_max_int', type=int, default=99)
    parser.add_argument('--sample_min_float', type=float, default=0.0)
    parser.add_argument('--sample_max_float', type=float, default=99.9)
    parser.add_argument('--float', type=bool, default=False)
    parser.add_argument('--use_words', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--freeze_embedder', type=bool, default=False)
    parser.add_argument('--context_units', type=bool, default=False)
    parser.add_argument('--log_filename', type=str, default='decoding_units.log')
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
    
    n_training_examples = args.training_examples
    n_test_examples = args.test_examples
    
    if args.context_units == True:
        task = 'Context_Units'
    else:
        task = 'Units'
    
    phase_message = 'Begin generating dataset.'
    report_phase(phase_message)
    
    units_path = "text-summarization-number-probing/units_processing/units.txt"
    data_path = "text-summarization-number-probing/units_processing/context_units.csv"
    
    training_dataset, test_dataset = generate_data(
        tokenizer, device, sample_min, sample_max,
        n_training_examples, n_test_examples, task,
        use_word_format=args.use_words,
        units_loc=units_path, data_loc=data_path)
    
    if args.embedding_model in ('Pegasus', 'T5', 'SSR', 'ProphetNet'):
        start_token_length = 0
    elif args.embedding_model in ('Bart', 'DistilBart', 'UniLM'):
        start_token_length = 1
#     else:
#         raise ValueError('Error: --embedding_model must be a valid model type.')
    
    padded_seq_len = training_dataset[0][0]['input_ids'].size()[-1] - 1 \
        - start_token_length
    output_dim = training_dataset[0][1].size()[-1]
    print('Output_dim:', output_dim)
    
    print('Padded seq len:', padded_seq_len)
    
    if args.embedding_model == 'UniLM':
        training_batch_size = 1
    else:
        training_batch_size = 64
    
    training_dataloader = DataLoader(training_dataset, 
                                     batch_size=training_batch_size, 
                                     shuffle=True)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=1, 
                                 shuffle=True)
    
    phase_message = 'Completed generating dataset.'
    report_phase(phase_message)
    
    embedding_model = get_embedding_model(model_name)
    
    if args.freeze_embedder:
        freeze_module(embedding_model, 'Pegasus')
    embedding_model = embedding_model.to(device)
    
    if args.context_units:
        dm = ContextUnitsModel(embedding_model, 
                               output_dim, 
                               padded_seq_len, 
                               args.hidden_dim).to(device)
    else:
        dm = UnitsModel(embedding_model, 
                        output_dim, 
                        padded_seq_len,
                        args.hidden_dim).to(device)

    phase_message = 'Model set up.'
    report_phase(phase_message)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # hyperparameters per Wallace et al. (2019) code
    if args.freeze_embedder:
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, dm.parameters()), 
                                    lr=args.lr, 
                                    momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(dm.parameters(), lr=args.lr, momentum=args.momentum)
    
    EPOCHS = args.epochs
    
    phase_message = 'Begin training.'
    report_phase(phase_message)
    
    epoch_number = 0
    
    for epoch in range(EPOCHS):
        epoch_message = 'Begin epoch {n}'.format(n=epoch_number + 1)
        report_phase(epoch_message)

        # Make sure gradient tracking is on, and do a pass over the data
        dm.train(True)
        avg_loss, continuing_loss, acc = train_epoch(
            epoch_number, training_dataloader, dm, loss_fn, 
            optimizer, output_dim)
        
        phase_message = f"End of epoch average batch loss: {avg_loss}"
        report_phase(phase_message)
        phase_message = f"End of epoch last loss: {continuing_loss}"
        report_phase(phase_message)
        phase_message = f"Epoch accuracy: {acc}"
        report_phase(phase_message)
        
        epoch_number += 1
        
    # temporary: save last version of model
    # TODO: reimplement to save best version
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"model_{timestamp}_{epoch_number}"
    torch.save(dm.state_dict(), model_path)
    
    # testing and metrics
    message = 'Training finished.'
    report_phase(message)
    
    dm.eval()
    
    message = 'Begin evaluation.'
    report_phase(message)
    accuracy = evaluate(dm, test_dataloader)
    message = f"Test accuracy: {accuracy}"
    report_phase(message)
