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

logging.basicConfig(filename='pegasus_max_number.log', level=logging.INFO)

import torch
from torch.utils.data import DataLoader

from torchmetrics import Accuracy

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from model import MaxProbingModel
from generate_data import generate_data


def train_epoch(idx, training_data_loader, model, loss_function, optimizer):
    batch_loss = 0.0
    continuing_loss = 0.0
    
    accuracy = Accuracy(num_classes=5)
    
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
        
        if i % 250 == 249:
            batch_loss = continuing_loss / 250
            n = i + 1
            loss_message = f"-- Batch {n} loss: {batch_loss}"
            print(loss_message)
            logging.info(loss_message)
            accuracy_message = f"-- Batch {n} accuracy: {batch_accuracy}"
            print(accuracy_message)
            logging.info(accuracy_message)
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


def report_phase(message):
    timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    formatted_message = f"{timestamp} | {message}"
    print(formatted_message)
    logging.info(formatted_message)

    
# to implement: calculate metrics
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_examples', type=int, default=1000)
    parser.add_argument('--test_examples', type=int, default=100)
    parser.add_argument('--sample_min_int', type=int, default=0)
    parser.add_argument('--sample_max_int', type=int, default=99)
    parser.add_argument('--sample_min_float', type=float, default=0.0)
    parser.add_argument('--sample_max_float', type=float, default=99.9)
    parser.add_argument('--float', type=bool, default=False)
    parser.add_argument('--use_words', type=bool, default=False)
    args = parser.parse_args()
    
    model_name = "google/pegasus-xsum"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    if args.float:
        sample_min = args.sample_min_float
        sample_max = args.sample_max_float
    else:
        sample_min = args.sample_min_int
        sample_max = args.sample_max_int
    
    n_training_examples = args.training_examples
    n_test_examples = args.test_examples
    
    phase_message = 'Begin generating dataset.'
    report_phase(phase_message)
    
    training_dataset, test_dataset = generate_data(
        tokenizer, device, sample_min, sample_max,
        n_training_examples, n_test_examples,
        use_word_format=args.use_words)
    
    training_dataloader = DataLoader(training_dataset, 
                                     batch_size=64, 
                                     shuffle=True)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=1, 
                                 shuffle=True)
    
    phase_message = 'Completed generating dataset.'
    report_phase(phase_message)
    
    pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)
    pegasus_model = pegasus_model.to(device)
    
    mpm = MaxProbingModel(pegasus_model).to(device)

    phase_message = 'Model set up.'
    report_phase(phase_message)
    
    # This choice of loss mirrors Wallace et al's (2019) code.
    # From the original paper:
    # "We use the negative log-likelihood of the maximum number as the loss function."
    # PyTorch's CrossEntropyLoss applies softmax along with the negative log-likelihood, as described in the paper.
    loss_fn = torch.nn.CrossEntropyLoss()

    # hyperparameters per Wallace et al. (2019) code
    optimizer = torch.optim.SGD(mpm.parameters(), lr=0.01, momentum=0.5)
    
    EPOCHS = 10
    
    phase_message = 'Begin training.'
    report_phase(phase_message)
    
    epoch_number = 0
    
    for epoch in range(EPOCHS):
        epoch_message = 'Begin epoch {n}'.format(n=epoch_number + 1)
        report_phase(epoch_message)

        # Make sure gradient tracking is on, and do a pass over the data
        mpm.train(True)
        avg_loss, continuing_loss, acc = train_epoch(epoch_number,
                                                training_dataloader,
                                                mpm,
                                                loss_fn, 
                                                optimizer)
        
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
    torch.save(mpm.state_dict(), model_path)
        
    # testing and metrics
    message = 'Training finished.'
    report_phase(message)
    message = 'Begin evaluation.'
    report_phase(message)
    accuracy = evaluate(mpm, test_dataloader)
    message = f"Test accuracy: {accuracy}"
    report_phase(message)
