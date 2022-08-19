# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import torch

from datasets import load_dataset

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')

results = []
for item in dataset['test']:
    doc = item['document']
    tokenized = tokenizer(doc, return_tensors='pt', truncation=True, max_length=128)
    decoder_tokenized = tokenizer('<s>', return_tensors='pt')
    tokenized['decoder_input_ids'] = decoder_tokenized['input_ids']
    last_word = None
    while True: # TODO: replace with maximum output length allowed by model decoder
        if last_word is not None:
            # TODO: determine correct reference to output dimensionality and update
            tokenized['decoder_input_ids'] = torch.cat([tokenized['decoder_input_ids'], last_word])
        output = model(**tokenized)
        last_word = output[-1]
        if last_word == '<eos>': # TODO: replace with correct call for '<eos>' index
            break
    # TODO: implement conversion from final decoder_input_ids
    out_sequence = decoder_output_to_word(tokenized['decoder_input_ids'])
    results.append((doc, out_sequence))
    
for (a, b) in results:
    print(a)
    print('-->')
    print(b)
