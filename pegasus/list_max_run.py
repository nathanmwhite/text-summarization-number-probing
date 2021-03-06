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

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from torchmetrics import Accuracy

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from .model import MaxProbingModel, report_phase, freeze_module
from .generate_data import generate_data
from .util import check_arguments, get_model_name, get_tokenizer, get_embedding_model
from .early_stopping import Early_Stopping

def train_epoch(idx, training_data_loader, model, loss_function, optimizer, clip_norm):
    batch_loss = 0.0
    continuing_loss = 0.0
    total_loss = 0.0
    
    accuracy = Accuracy(num_classes=5)
    
    for i, data_batch in enumerate(training_data_loader):
        inputs, labels = data_batch
        
        optimizer.zero_grad()
        
        # testing only
#         print('Inputs size:', inputs['input_ids'].size())
#         print('Inputs:', inputs['input_ids'])
        
        outputs = model(inputs)
        
        # debug only
#         report_phase('Outputs size:', outputs.size())
#        report_phase(outputs.size())
#         report_phase('Outputs:', outputs)
#         report_phase('Labels size:', labels.size())
#        report_phase(labels.size())
#         report_phase('Labels:', labels)
        
        loss = loss_function(outputs, labels)
        
        loss.backward()
        
        #label_int_tensor = torch.argmax(labels, axis=-1)
        
        # testing only
        #print(label_int_tensor.device)
        #print(outputs.device)
        
        # torchmetrics implementation requires transfer to CPU
        #labels_cpu = label_int_tensor.to("cpu")
        labels_cpu = labels.to("cpu")
        outputs_cpu = outputs.to("cpu")
        
        batch_accuracy = accuracy(outputs_cpu, labels_cpu)
        
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
            accuracy_message = f"-- Batch {n} accuracy: {batch_accuracy}"
            print(accuracy_message)
            logging.info(accuracy_message)
            continuing_loss = 0.0
            
    return batch_loss, continuing_loss, total_loss, accuracy.compute()


def evaluate(model, eval_dataloader):
    model.eval()
    accuracy = Accuracy()
    
    for i, data_point in enumerate(eval_dataloader):
        inputs, labels = data_point
        
        output = model(inputs)
        
        #label_int_tensor = torch.argmax(labels, axis=-1)
        
        # torchmetrics implementation requires transfer to CPU
        #labels_cpu = label_int_tensor.to("cpu")
        labels_cpu = labels.to("cpu")
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
    parser.add_argument('--sample_min_float', type=int, default=0)
    parser.add_argument('--sample_max_float', type=int, default=99)
    parser.add_argument('--float', type=bool, default=False)
    parser.add_argument('--use_words', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--freeze_embedder', type=bool, default=False)
    parser.add_argument('--log_filename', type=str, default='max_number.log')
    parser.add_argument('--trial_number', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--early_stopping', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--clip_norm', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=5)
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
    
    n_training_examples = args.training_examples
    n_test_examples = args.test_examples
    
#     phase_message = 'Begin generating dataset.'
#     report_phase(phase_message)
    
    training_dataset, test_dataset = generate_data(
        tokenizer, device, sample_min, sample_max,
        n_training_examples, n_test_examples, 'ListMax',
        use_word_format=args.use_words,
        float_=args.float)

    if args.embedding_model in ('Pegasus', 'T5', 'SSR', 'ProphetNet'):
        start_token_length = 0
    elif args.embedding_model in ('Bart', 'DistilBart', 'UniLM', 'Bert', 'Random'):
        start_token_length = 1
#     else:
#         raise ValueError('Error: --embedding_model must be a valid model type.')
    
    padded_seq_len = training_dataset[0][0]['input_ids'].size()[-1] - 1 \
        - start_token_length
    
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
    
#     phase_message = 'Completed generating dataset.'
#     report_phase(phase_message)
    
    embedding_model = get_embedding_model(model_name, args.trained)

    if args.freeze_embedder:
        freeze_module(embedding_model, args.embedding_model)
    embedding_model = embedding_model.to(device)
    
    mpm = MaxProbingModel(embedding_model, padded_seq_len, hidden_dim=args.hidden_dim).to(device)

#     phase_message = 'Model set up.'
#     report_phase(phase_message)
    
    # This choice of loss mirrors Wallace et al's (2019) code.
    # From the original paper:
    # "We use the negative log-likelihood of the maximum number as the loss function."
    # PyTorch's CrossEntropyLoss applies softmax along with the negative log-likelihood, 
    #     as described in the paper.
    # however, the model already has a log_softmax layer applied, so this should be
    #     NLLLoss
    #loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.NLLLoss()

    # hyperparameters per Wallace et al. (2019) code
    if args.freeze_embedder:
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, mpm.parameters()),
                                    lr=args.lr, 
                                    momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(mpm.parameters(), lr=args.lr, momentum=args.momentum)
    
    EPOCHS = args.epochs
    
#     phase_message = 'Begin training.'
#     report_phase(phase_message)

    early_stopping = Early_Stopping(min_delta=0.0, patience=args.patience)
    
    epoch_number = 0
    
    for epoch in range(EPOCHS):
#         epoch_message = 'Begin epoch {n}'.format(n=epoch_number + 1)
#         report_phase(epoch_message)

        # Make sure gradient tracking is on, and do a pass over the data
        mpm.train(True)
        avg_loss, continuing_loss, total_loss, acc = train_epoch(epoch_number,
                                                                 training_dataloader,
                                                                 mpm,
                                                                 loss_fn, 
                                                                 optimizer,
                                                                 args.clip_norm)
        
#         phase_message = f"End of epoch average batch loss: {avg_loss}"
#         report_phase(phase_message)
#         phase_message = f"End of epoch last loss: {continuing_loss}"
#         report_phase(phase_message)
#         phase_message = f"Epoch accuracy: {acc}"
#         report_phase(phase_message)

        if args.early_stopping:
            early_stopping(total_loss)
            if early_stopping.early_stopping == True:
                message = f'Early stopping of training at epoch {epoch_number}.'
                report_phase(message)
                break
        
        epoch_number += 1
        
    # temporary: save last version of model
    # TODO: reimplement to save best version
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     model_path = f"model_{timestamp}_{epoch_number}"
#     torch.save(mpm.state_dict(), model_path)
        
    # testing and metrics
#     message = 'Training finished.'
#     report_phase(message)
#     message = 'Begin evaluation.'
#     report_phase(message)
    mpm.eval()
    with torch.no_grad():
        accuracy = evaluate(mpm, test_dataloader)
    hyperparam_set = ('ListMax trial',
                      args.embedding_model,
                      args.training_examples,
                      args.lr,
                      args.momentum,
                      args.epochs,
                      args.trial_number)
                      
    message = f"Model hyperparameters: " + ' | '.join(str(w) for w in hyperparam_set)
    report_phase(message)
    message = f"Test accuracy: {accuracy}"
    report_phase(message)
