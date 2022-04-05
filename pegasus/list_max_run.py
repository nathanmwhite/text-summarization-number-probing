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

from model import MaxProbingModel
# TODO: implement and import data generation code here

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
            
    return prior_loss

# to implement: calculate metrics
if __name__ == '__main__':
    model_name = "google/pegasus-xsum"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    
    mpm = MaxProbingModel(pegasus_model).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(mpm.parameters(), lr=0.01, momentum=0.5) # hyperparameters per Wallace et al. (2019) code
    
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
        model.train(True)
        avg_loss = train_one_epoch(epoch_number)
        
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

        epoch_number += 1
