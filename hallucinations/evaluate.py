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

from ..pegasus.util import get_tokenizer, get_embedding_model

# TODO: create support for evaluating on the final version of malo_cleaned

from torch.utils.data import Dataset, DataLoader


class GenerationDataset(Dataset):
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
    tokenized = tokenizer(data_in, return_tensors='pt', padding=True)
    dataset = GenerationDataset(tokenized).to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
