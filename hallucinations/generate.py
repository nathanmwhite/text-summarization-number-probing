# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import argparse

import string

import torch

from datasets import load_dataset

#from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import T5ForConditionalGeneration, ProphetNetForConditionalGeneration

from ..pegasus.util import get_model_name, get_tokenizer, get_embedding_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def is_a_number(sequence):
    for i, c in enumerate(sequence):
        if c in string.digits:
            continue
        elif c in '-,.' and i > 0:
            continue
        else:
            return False
    return True


def get_numbers(data_line):
    word_sequence = data_line.split(' ')
    found = []
    for word in word_sequence:
        if is_a_number(word):
            found.append(word)
    return found


def values_shared_check(group1, group2):
    set1 = set(group1)
    set2 = set(group2)
    if set1.intersection(set2) == set() and set2 != set():
        return False
    else:
        return True


def generate_results(tokenizer, model, dataset, task_prefix, values_shared):
    results = []
    for item in dataset['test']:
        doc = item['document']
        
        doc_numbers = get_numbers(doc)

        if task_prefix and type(model) == T5ForConditionalGeneration:
            batch_result = tokenizer.prepare_seq2seq_batch(src_texts='summarize: ' + doc, return_tensors='pt')
        else:
            batch_result = tokenizer.prepare_seq2seq_batch(src_texts=doc, return_tensors='pt')
        #print(batch_result)
        batch_result_out = {}
        for k in batch_result.keys():
            print(type(model))
            if k == 'token_type_ids' and type(model) == ProphetNetForConditionalGeneration:
                continue
            else:
                batch_result_out[k] = batch_result[k].to(device)

        out = model.generate(**batch_result_out)
        #print(out)
        out_sequence = tokenizer.batch_decode(out)
        
        out_numbers = get_numbers(out_sequence[0])
        
        if values_shared and values_shared_check(doc_numbers, out_numbers) == False:
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
    parser.add_argument('--task_prefix', type=bool, default=False)
    parser.add_argument('--values_shared', type=bool, default=False)
    args = parser.parse_args()
    
    model_name = get_model_name(args.embedding_model)
    tokenizer = get_tokenizer(model_name)
    model = get_embedding_model(model_name).to(device)
    
    dataset = load_dataset('xsum', cache_dir='./models')

    results = generate_results(tokenizer, model, dataset, args.task_prefix, args.values_shared)
    record_results(results)