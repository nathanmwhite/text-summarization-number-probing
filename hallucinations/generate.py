# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import argparse

import torch

from datasets import load_dataset

#from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from .util import get_model_name, get_tokenizer, get_embedding_model


def generate_results(tokenizer, model, dataset):
    results = []
    for item in dataset['test']:
        doc = item['document']
        #tokenized = tokenizer(doc, return_tensors='pt', truncation=True, max_length=128)
        batch_result = tokenizer.prepare_seq2seq_batch(src_texts=doc)
        out = model.generate(**batch_result)
        out_sequence = tokenizer.batch_decode(out)
        results.append((doc, out_sequence))
    return results


def record_results(results):
    for (a, b) in results:
        print(a)
        print('-->')
        print(b)
        print('-----')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_model', type=str, default='Pegasus')
    args = parser.parse_args()

    model_name = get_model_name(args.embedding_model)
    tokenizer = get_tokenizer(model_name)
    model = get_embedding_model(model_name)
    
    dataset = load_dataset('xsum', cache_dir='./models')

    results = generate_results(tokenizer, model, dataset)
    record_results(results)
