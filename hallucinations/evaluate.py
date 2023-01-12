# Text summarization number probing
# Original code Copyright © 2023 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2023 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

from transformers import PegasusForConditionalGeneration 
from transformers import T5ForConditionalGeneration
from transformers import BartForConditionalGeneration
from transformers import ProphetNetForConditionalGeneration

from torch.utils.data import Dataset, DataLoader

from ..pegasus.util import get_tokenizer, get_embedding_model
from .util import get_numbers, check_numerical

# TODO: create support for evaluating on the final version of malo_cleaned


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
        self._input_ids.to(device)
        self._attention_mask.to(device)

        
def create_eval_dataloader(data_in, batch_size, device):
    """
    create_eval_dataloader : creates dataloader for use in evaluating
        hallucination in the context of generating summaries.
    @param data_in (List[str]) : list of data points to serve as input
        to generate summaries; note: should not be List[List[str]],
        as this will cause the torch-based tokenizer to treat all the data
        as a single document to embed
    @param batch_size (int) : integer indicating batch size
    @param device (torch.device) : the device to use in summary generation
    returns : torch DataLoader containing the batched data to use in
        generating summaries to evaluate hallucinations.
    """
    # assert(type(data_in) == List)
    # assert(type(data_in[0]) == str)
    tokenized = tokenizer(data_in, return_tensors='pt', padding=True)
    dataset = GenerationDataset(tokenized).to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def evaluate(model, dataloader, input_data):
    model.eval()
    
    outputs = []
    for i, data_point in enumerate(dataloader):
        print(f'Processing batch {i}')
        inputs = data_point
        
        output = model.generate(**inputs)
        
        for item in output:
            string_ = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(item))
            outputs.append(string_)
            
        # next step is to implement metric for matching numerical values
        metrics = check_numerical(input_data, outputs)
                
    return metrics
