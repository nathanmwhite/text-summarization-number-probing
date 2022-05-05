# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import os

from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration
from ..s2s_ft.tokenization_unilm import UnilmTokenizer
from ..s2s_ft.modeling_decoding import BertConfig, BertForSeq2SeqDecoder


def check_arguments(args):
    if args.sample_min_int >= args.sample_max_int:
        return ValueError('Argument sample_min_int is too high compared with sample_max_int.')
    if args.sample_min_float >= args.sample_max_float:
        return ValueError('Argument sample_min_float is too high compared with sample_max_float.')
    if args.float and args.use_words:
        return ValueError('Arguments float and use_words cannot both be True.')
    if args.epochs <= 0:
        return ValueError('Argument epochs has an invalid value. It must be greater than 0.')
    if args.training_examples <= 0:
        return ValueError('Argument training_examples has an invalid value. It must be greater than 0.')
    if args.test_examples <= 0:
        return ValueError('Argument test_examples has an invalid value. It must be greater than 0.')

    
def obtain_units(source_loc):
    with open(source_loc, 'r') as f:
        unit_strings = f.readlines()
    units = (tuple(line.rstrip().split('/')) for line in unit_strings)
    return units


MODEL_NAME_MAP = {'Pegasus': "google/pegasus-xsum",
                  'T5': "t5-base",
                  'SSR': "microsoft/ssr-base",
                  'Bart': "facebook/bart-base",
                  'DistilBart': "sshleifer/distilbart-xsum-12-6",
                  'ProphetNet': "microsoft/prophetnet-large-uncased",
                  'UniLM': "unilm2-base-uncased"}


def get_model_name(model_type):
    try:
        name = MODEL_NAME_MAP[model_type]
    except KeyError as e:
        print('Invalid embedding_model specified: {model_type}')
        raise
    return name


def get_tokenizer(model_name):
    if model_name == MODEL_NAME_MAP['Pegasus']:
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
    elif model_name in (MODEL_NAME_MAP['T5'], MODEL_NAME_MAP['SSR']):
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif model_name in (MODEL_NAME_MAP['Bart'], MODEL_NAME_MAP['DistilBart']):
        tokenizer = BartTokenizer.from_pretrained(model_name)
    elif model_name == MODEL_NAME_MAP['ProphetNet']:
        tokenizer = ProphetNetTokenizer.from_pretrained(model_name)
    elif model_name == MODEL_NAME_MAP['UniLM']:
        vocab_path = os.path.abspath(os.path.join('..', model_name, model_name + '-vocab.txt'))
        tokenizer = UnilmTokenizer.from_pretrained(vocab_path)
    return tokenizer

                        
def get_embedding_model(model_name):
    if model_name == MODEL_NAME_MAP['Pegasus']:
        embedding_model = PegasusForConditionalGeneration.from_pretrained(model_name)
    elif model_name in (MODEL_NAME_MAP['T5'], MODEL_NAME_MAP['SSR']):
        embedding_model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif model_name in (MODEL_NAME_MAP['Bart'], MODEL_NAME_MAP['DistilBart']):
        embedding_model = BartForConditionalGeneration.from_pretrained(model_name)
    elif model_name == MODEL_NAME_MAP['ProphetNet']:
        # TODO: test if AutoModelForConditionalGeneration works; 
        #  default listed is AutoModelForSeq2SeqLM
        embedding_model = ProphetNetForConditionalGeneration.from_pretrained(model_name)
    elif model_name == MODEL_NAME_MAP['UniLM']:
        VOCAB_SIZE = 30522
        TYPE_VOCAB_SIZE = 2
        config_ = BertConfig(rel_pos_bins=0, 
                             vocab_size_or_config_json_file=VOCAB_SIZE, 
                             type_vocab_size=TYPE_VOCAB_SIZE)
        model_path = os.path.abspath(os.path.join('..', model_name))
        embedding_model = BertForSeq2SeqDecoder.from_pretrained(model_path, config=config_, search_beam_size=2)
    return embedding_model
