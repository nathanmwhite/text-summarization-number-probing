# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

from datetime import datetime

import logging

logging.basicConfig(filename='pegasus_max_number.log', level=logging.INFO)

import torch
from torch.utils.data import DataLoader

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from model import MaxProbingModel
from generate_data import generate_data

def train_epoch(idx, training_data_loader, model, loss_function, optimizer):
    batch_loss = 0.0
    continuing_loss = 0.0
    
    for i, data_batch in enumerate(training_data_loader):
        inputs, labels = data_batch
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = loss_function(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        continuing_loss += loss.item()
        
        if i % 250 == 249:
            batch_loss = continuing_loss / 250
            loss_message = '-- Batch {n} loss: {loss}'.format(n = i + 1, loss = batch_loss)
            print(loss_message)
            logging.info(loss_message)
            continuing_loss = 0.0
            
    return batch_loss

# to implement: calculate metrics
if __name__ == '__main__':
    model_name = "google/pegasus-xsum"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    sample_min = 0
    sample_max = 99
    
    n_training_examples = 1000
    n_test_examples = 100
    
    phase_message = 'Begin generating dataset.'
    print(phase_message)
    logging.info(phase_message)
    
    training_dataset, test_dataset = generate_data(tokenizer,
                                                   device,
                                                   sample_min,
                                                   sample_max,
                                                   n_training_examples,
                                                   n_test_examples)
    
    training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    phase_message = 'Completed generating dataset.'
    print(phase_message)
    logging.info(phase_message)
    
    pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    
    mpm = MaxProbingModel(pegasus_model).to(device)

    phase_message = 'Model set up.'
    print(phase_message)
    logging.info(phase_message)
    
    # This choice of loss mirrors Wallace et al's (2019) code.
    # From the original paper:
    # "We use the negative log-likelihood of the maximum number as the loss function."
    # PyTorch's CrossEntropyLoss applies softmax along with the negative log-likelihood, as described in the paper.
    loss_fn = torch.nn.CrossEntropyLoss()

    # hyperparameters per Wallace et al. (2019) code
    optimizer = torch.optim.SGD(mpm.parameters(), lr=0.01, momentum=0.5)
    
    EPOCHS = 10
    
    current_timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    timestamp_message = '{timestamp} | Begin training'.format(timestamp=current_timestamp)
    print(timestamp_message)
    logging.info(timestamp_message)
    epoch_number = 0
    
    for epoch in range(EPOCHS):
        epoch_message = 'Epoch {n}:'.format(n=epoch_number + 1)
        print(epoch_message)
        logging.info(epoch_message)

        # Make sure gradient tracking is on, and do a pass over the data
        mpm.train(True)
        avg_loss = train_epoch(epoch_number, training_dataloader, mpm, loss_fn, optimizer)
        
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(mpm.state_dict(), model_path)

        epoch_number += 1
        
# TODO: implement testing and metrics
