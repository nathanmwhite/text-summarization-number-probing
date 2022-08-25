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
from transformers import BertTokenizer, BertModel
from transformers import BertConfig as TrBertConfig
from .model import RandomEmbeddingModel
from ..s2s_ft.tokenization_unilm import UnilmTokenizer
from ..s2s_ft.modeling_decoding import BertForSeq2SeqDecoder
from ..s2s_ft.modeling_decoding import BertConfig as S2SBertConfig

from .model import report_phase

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
    units = tuple(tuple(line.rstrip().split('/')) for line in unit_strings)
    return units


MODEL_NAME_MAP = {'Pegasus': "google/pegasus-xsum",
                  'T5': "t5-base",
                  'T5-CDM': "flax-community/t5-base-cnn-dm", # unknown if same tokenizer is appropriate
                  'SSR': "microsoft/ssr-base", # no equivalent for SSR
                  'Bart': "facebook/bart-base",
                  'Bart-XSum': "facebook/bart-large-xsum",
                  'DistilBart': "sshleifer/distilbart-xsum-12-6",
                  'ProphetNet': "microsoft/prophetnet-large-uncased",
                  'ProphetNet-CDM': "microsoft/prophetnet-large-uncased-cnndm", 
                  'UniLM': "unilm2-base-uncased",
                  'Bert': "bert-base-uncased",
                  'Random': "random-vectors"}


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
    elif model_name in (MODEL_NAME_MAP['T5'], MODEL_NAME_MAP['SSR'], MODEL_NAME_MAP['T5-CDM']):
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif model_name in (MODEL_NAME_MAP['Bart'], MODEL_NAME_MAP['DistilBart'], MODEL_NAME_MAP['Bart-XSum']):
        tokenizer = BartTokenizer.from_pretrained(model_name)
    elif model_name in (MODEL_NAME_MAP['ProphetNet'], MODEL_NAME_MAP['ProphetNet-CDM']):
        tokenizer = ProphetNetTokenizer.from_pretrained(model_name)
    elif model_name == MODEL_NAME_MAP['UniLM']:
        vocab_path = os.path.abspath(os.path.join('text-summarization-number-probing', 
                                                  model_name, 
                                                  model_name + '-vocab.txt'))
        tokenizer = UnilmTokenizer.from_pretrained(vocab_path)
    elif model_name == MODEL_NAME_MAP['Bert']:
        tokenizer = BertTokenizer.from_pretrained(model_name)
    elif model_name == MODEL_NAME_MAP['Random']:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_MAP['Bert'])
    return tokenizer

                        
def get_embedding_model(model_name, trained=True):
    if model_name == MODEL_NAME_MAP['Pegasus']:
        embedding_model = PegasusForConditionalGeneration.from_pretrained(model_name)
    elif model_name in (MODEL_NAME_MAP['T5'], MODEL_NAME_MAP['SSR'], MODEL_NAME_MAP['T5-CDM']):
        embedding_model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif model_name in (MODEL_NAME_MAP['Bart'], MODEL_NAME_MAP['DistilBart'], MODEL_NAME_MAP['Bart-XSum']):
        embedding_model = BartForConditionalGeneration.from_pretrained(model_name)
    elif model_name in (MODEL_NAME_MAP['ProphetNet'], MODEL_NAME_MAP['ProphetNet-CDM']):
        # TODO: test if AutoModelForConditionalGeneration works; 
        #  default listed is AutoModelForSeq2SeqLM
        embedding_model = ProphetNetForConditionalGeneration.from_pretrained(model_name)
        # use_cache=False to avoid clash with gradient checkpointing used by ProphetNet
        embedding_model.config.use_cache = False
    elif model_name == MODEL_NAME_MAP['UniLM']:
        VOCAB_SIZE = 30522
        TYPE_VOCAB_SIZE = 2
        config_ = S2SBertConfig(rel_pos_bins=0, 
                             vocab_size_or_config_json_file=VOCAB_SIZE, 
                             type_vocab_size=TYPE_VOCAB_SIZE)
        model_path = os.path.abspath(os.path.join('text-summarization-number-probing', model_name))
        embedding_model = BertForSeq2SeqDecoder.from_pretrained(model_path, 
                                                                config=config_, 
                                                                search_beam_size=2)
    elif model_name == MODEL_NAME_MAP['Bert']:
        if trained:
            report_phase('Loading pretrained Bert model.')
            embedding_model = BertModel.from_pretrained(model_name)
        else:
            report_phase('Creating new untrained Bert model.')
            config_ = TrBertConfig()
            embedding_model = BertModel(config_)
    elif model_name == MODEL_NAME_MAP['Random']:
        embedding_model = RandomEmbeddingModel()
           
    return embedding_model
