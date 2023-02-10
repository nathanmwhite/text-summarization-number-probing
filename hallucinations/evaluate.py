# Text summarization number probing
# Original code Copyright © 2023 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2023 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import argparse

import logging

import numpy as np

from transformers import PegasusForConditionalGeneration 
from transformers import T5ForConditionalGeneration
from transformers import BartForConditionalGeneration
from transformers import ProphetNetForConditionalGeneration

import torch

from torch.utils.data import Dataset, DataLoader

from ..pegasus.model import report_phase
from ..pegasus.util import get_model_name, get_tokenizer, get_embedding_model
from .util import check_numerical, get_numbers, load_malo_data, retokenize


class GenerationDataset(Dataset):
    """
    GenerationDataset : dataset to serve in evaluation containing
        tokenized input_ids and attention_mask elements for generating
        summarization text to evaluate hallucination.
    """
    def __init__(self, input_data):
        self._input_ids = input_data['input_ids']
        self._attention_mask = input_data['attention_mask']
        
    def __len__(self):
        return len(self._input_ids)
    
    def __getitem__(self, index):
        ids = self._input_ids[index]
        mask = self._attention_mask[index]
        datapoint = {'input_ids': ids, 'attention_mask': mask}
        
        return datapoint
    
    def to(self, device):
        self._input_ids = self._input_ids.to(device)
        self._attention_mask = self._attention_mask.to(device)
        
        # return self so that .to functionality of object is consistent with PyTorch pattern
        return self

        
def create_eval_dataloader(data_in, batch_size, tokenizer, device):
    """
    create_eval_dataloader : creates dataloader for use in evaluating
        hallucination in the context of generating summaries.
    @param data_in (List[str]) : list of data points to serve as input
        to generate summaries; note: should not be List[List[str]],
        as this will cause the torch-based tokenizer to treat all the data
        as a single document to embed
    @param batch_size (int) : integer indicating batch size
    @param tokenizer (Tokenizer) : tokenizer to use for tokenization of data
    @param device (torch.device) : the device to use in summary generation
    returns : torch DataLoader containing the batched data to use in
        generating summaries to evaluate hallucinations.
    """
    # assert(type(data_in) == List)
    # assert(type(data_in[0]) == str)
    tokenized = tokenizer(data_in, return_tensors='pt', padding=True)
    
    # debug
    #report_phase(len(tokenized['input_ids']))
        
    dataset = GenerationDataset(tokenized).to(device)
    
    # debug
    #report_phase(dataset[0])
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def evaluate(model, dataloader, input_data):
    """
    evaluate : function that evaluates models for numerical hallucination.
    @param model (torch.nn.Module) : model to evaluate
    @param dataloader (torch.utils.data.DataLoader) : dataloader containing
        the batched input data
    @param input_data (List[str]) : list containing the input data as raw text
        to evaluate outputs against
    returns : List[tuple] containing 3-tuples with results for each input item,
        where the tuple contains the following values:
        1) the number of numerical items found only in the input string
        2) the number of numerical items found only in the generated output summary
        3) the number of numerical items shared between the two.
    """
    model.eval()
    
    retokenized_outputs = []
    
    for i, data_point in enumerate(dataloader):
        print(f'Processing batch {i}')

        inputs = data_point
        
        output = model.generate(**inputs)
        
        output_strings = []
        for item in output:
            string_ = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(item))
            output_strings.append(string_)
            
        retokenized_outputs += retokenize(output_strings)
        
    # debug
    report_phase(retokenized_outputs[0])
    report_phase(retokenized_outputs[-1])
    report_phase(len(retokenized_outputs))
            
    metrics = check_numerical(input_data, retokenized_outputs)
                
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_model', type=str, default='Pegasus')
    parser.add_argument('--malo_datapath', type=str, default='./text-summarization-number-probing/hallucinations/malo_cleaned.txt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_filename', type=str, default='evaluate.log')
    args = parser.parse_args()
    
    logging.basicConfig(filename=args.log_filename, level=logging.INFO)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = get_model_name(args.embedding_model)
    
    tokenizer = get_tokenizer(model_name)
    
    model = get_embedding_model(model_name, trained=True).to(device)
    
    # debug
    #report_phase(args.malo_datapath)
    
    data_in = load_malo_data(args.malo_datapath)
    
    # debug
    #report_phase(len(data_in))
    
    dataloader = create_eval_dataloader(data_in, args.batch_size, tokenizer, device)
    
    results = evaluate(model, dataloader, data_in)
    
    summed_results = np.sum(results, axis=0)
                      
    message = f"Generating model: {args.embedding_model}"
    report_phase(message)
    message = f"Found only in input: {summed_results[0]}"
    report_phase(message)
    message = f"Hallucinated only in output: {summed_results[1]}"
    report_phase(message)
    message = f"Matched in both input and output: {summed_results[2]}"
    report_phase(message)
