# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import math
import random

from num2words import num2words

import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

from .util import obtain_units
from ..units_processing.retrieve_units import is_a_number as isnumeric

# TODO: Methodology from Wallace et al. (2019) for their probes:
# "Each list consists of values of similar magnitude in order to evaluate fine-grained comparisons"
# "We first pick a range (we vary the range in our experiments) and randomly shuffle the integers over it. We then split
#   80% of the numbers into a training set and 20% into a test set. We report the mean and standard
#   deviation across five different random shuffles for a particular range, using the exact same shuffles
#   across all embedding methods."
# "Numbers are provided as integers (“75”), single-word form (“seventy-five”), floats (“75.1”),
#   or negatives (“-75”). We consider positive numbers less than 100 for word-form numbers to avoid
#   multiple tokens."
# List maximum:
# "For the list maximum task, we first shuffle and split the data, putting 80% into a training pool of
#   numbers and 20% into a test pool. In initial experiments, we created the lists of five numbers by sampling uniformly over the
#   training/test pool. However, as the random samples will likely be spread out over the range, the numbers are easy to distinguish.
#   We instead create 100,000 training examples and 10,000 examples in the following manner.
#   We first sample a random integer from the training or test pool. Next, we sample from a Gaussian
#   with mean zero and variance equal to 0.01 times the total size of the range. Finally, we add the
#   random Gaussian sample to the random integer, and round to the nearest value in the pool. This forces the numbers to be nearby."
# Addition:
# "We create training/test splits for the addition task in the following manner. We first shuffle and split
#   an integer range, putting 80% into train and 20% into test. Next, we enumerate all possible pairs of
#   numbers in the two respective splits. When using large ranges such as [0,999], we sub-sample a random 10% of the training and test pairs.

# TODO: Implement Addition and Decoding data generation

# TODO: for List Maximum, enable float/negative representations

#sample_max = 99
#sample_min = 0
#num_train_examples = 100000
#num_test_examples = 10000

#datapoint_length = 5


# Define custom Dataset
class ProbingDataset(Dataset):
    def __init__(self, input_data, decoder_input_data, output_data):
        self._inputs = input_data
        self._decoder_inputs = decoder_input_data
        self._outputs = output_data
        
    def __len__(self):
        return len(self._outputs)
    
    def __getitem__(self, idx):
        input_ids = self._inputs['input_ids'][idx]
        attention_mask = self._inputs['attention_mask'][idx]
        decoder_input_ids = self._decoder_inputs[idx]
        _output = self._outputs[idx]

        _inputs = {'input_ids': input_ids, 
                   'attention_mask': attention_mask, 
                   'decoder_input_ids': decoder_input_ids}
        
        return _inputs, _output
    
    
class RangeProbingDataset(Dataset):
    def __init__(self, input_data, decoder_input_data, output_data_y1, output_data_y2):
        self._inputs = input_data
        self._decoder_inputs = decoder_input_data
        self._outputs_y1 = output_data_y1
        self._outputs_y2 = output_data_y2
        
    def __len__(self):
        return len(self._outputs_y1)
    
    def __getitem__(self, idx):
        input_ids = self._inputs['input_ids'][idx]
        attention_mask = self._inputs['attention_mask'][idx]
        decoder_input_ids = self._decoder_inputs[idx]
        _output_y1 = self._outputs_y1[idx]
        _output_y2 = self._outputs_y2[idx]

        _inputs = {'input_ids': input_ids, 
                   'attention_mask': attention_mask, 
                   'decoder_input_ids': decoder_input_ids}
        
        return _inputs, _output_y1, _output_y2
    
    
# their description does not specify what happens to the obtained value via the Gaussian process
# their code shows that the Gaussian is run five times per data point and appended
# this code follows the description in Wallace et al. (2019); their code has additional complications
#  and behavior not justified by their description
# TODO: revisit and reorganize training and test data processing into unified approach
def generate_data(tokenizer: PreTrainedTokenizer,
                  device: str,
                  sample_min: int, 
                  sample_max: int, 
                  num_training_examples: int, 
                  num_test_examples: int,
                  task: str,
                  datapoint_length: int=5,
                  use_word_format: bool=False,
                  units_loc: str=None,
                  data_loc: str=None):
    """
    generate_data : Function that generates training and test data for  
        the List Maximum task specified in Wallace et al. (2019).
    @param tokenizer (transformers.PreTrainedTokenizer) : Tokenizer to 
        use in dataset generation
    @param device (str) : either "cpu" or "cuda"
    @param sample_min (int) : Minimum of range to sample
    @param sample_max (int) : Maximum of range to sample
    @param num_training_examples (int) : Number of training examples to 
        generate
    @param num_test_examples (int) : Number of test examples to generate
    @param task (str) : Task to generate data for from among:
        ListMax, Decoding, Addition, Percent, Basis_Points, Units,
        Context_Units, Ranges, Orders
    @param datapoint_length (int) : Number of elements in each datapoint
    @param use_word_format (bool) : Indicates whether to convert an 
        integer into:
            word representation such as 37 --> "thirty-seven" (True), or
            keep as integer representation in a string as in 37 --> "37"
             (False)
    @param units_loc (str) : Indicates location of text file containing
        units as lines, with variants separated by slashes ('/'), where
        'Units' is task
    @param data_loc (str) : Indicates location of text file containing
        sentences with numbers and units identified as lines, where
        'Context_Units' is task
    returns : two ProbingDatasets: training, test datasets;
        for task 'Ranges', two RangeProbingDatasets: training, test
    """
    def generate_pools():
        # the definition from Wallace et al. could mean:
        # 1. they shuffled and split based on the integer numbers, or
        # 2. they shuffled and split based on string representations (unlikely because Gaussian wouldn't work then)
        # I conclude they shuffled and split based on the integer numbers, meaning different integers would be seen in training and testing
        #  --checked: This is in fact what they did in their code: numeracy/max.py, lines 143-149
        # prevents log(0) or log(negative); numbers such as 0 billion are unnatural anyway

        nonlocal sample_min
        if task == 'Orders' and sample_min <= 0:
            sample_min = 1
        data_range = np.asarray(range(sample_min, sample_max + 1))

        np.random.shuffle(data_range)
        split = math.floor(data_range.size * 0.8)
        training_pool = data_range[:split]
        test_pool = data_range[split:]

        return training_pool, test_pool

    def sample_gaussian(pool):
        sample_random_integer = np.random.choice(pool)
        scale = (sample_max - sample_min) * 0.01
        gaussian_sample = np.random.normal(scale=scale)
        add_result = sample_random_integer + gaussian_sample
        nearest_value = pool[np.argmin(np.abs(pool - add_result))]

        return nearest_value

    def sample_unit(units_data, singular=False):
        # logic: assumes units_data as series of (target_index, singular, plural) forms
        maximum = len(units_data) - 1
        idx = random.randint(0, maximum)
        target_index, singular_form, plural_form = units_data[idx]
        if singular:
            return singular_form, int(target_index)
        else:
            return plural_form, int(target_index)
    
    # TODO: evaluate whether to tease out two version for the different tasks
    def generation_loop(pool, num_examples, units=False, ranges=False, orders=False):
        assert(units == False or ranges == False or orders == False)
        assembled_data = []
        if units:
            assembled_units = []
            assembled_targets = []
        elif ranges:
            assembled_range_terms = []
        elif orders:
            assembled_orders = []
        for i in range(num_examples):
            datapoint = []
            if units:
                datapoint_units = []
                datapoint_targets = []
            elif ranges:
                assembled_range_terms.append(sample_terms(range_terms))
            elif orders:
                assembled_orders.append(sample_terms(order_terms))
            for j in range(datapoint_length):
                if units:
                    value = sample_gaussian(pool)
                    while value in datapoint:
                        value = sample_gaussian(pool)
                    singular = value == 1
                    unit, target_idx = sample_unit(units_data, singular)
                    datapoint.append(value)
                    datapoint_units.append(unit)
                    datapoint_targets.append(target_idx)
                else:
                    value = sample_gaussian(pool)
                    while value in datapoint:
                        value = sample_gaussian(pool)
                    datapoint.append(value)
                
            assembled_data.append(datapoint)
            if units:
                assembled_units.append(datapoint_units)
                assembled_targets.append(datapoint_targets)
        if units:
            return assembled_data, assembled_units, assembled_targets
        elif ranges:
            return assembled_data, assembled_range_terms
        elif orders:
            return assembled_data, assembled_orders
        return assembled_data
    
    # TODO: must handle multiword units with the last containing the unit
    # it will be more advantageous to have this data prepped in advance
    def extract_data(data_loc, units_loc):
        """
        extract_data : Function that generates training and test data derived
            from data sources that contain numbers with units.
        @param data_loc (str) : Path to data file stored as lines of sentences.
        @param units_loc (str) : Path to data file stored as lines of units.
        returns : TODO
        """
        units_data = obtain_units(units_loc)

        # raw_data is of format [sentence[tab]number[tab]target_unit_idx]
        with open(data_loc, 'r', encoding='iso-8859-1') as f:
            raw_data = f.readlines()

        data = tuple(tuple(line.rstrip().split('\t')) for line in raw_data)

        sentences, numerals, units = zip(*data)
        
        units = tuple(int(n) for n in units)
        
        return sentences, numerals, units
    
    
    def sample_terms(terms):
        idx = np.random.randint(len(terms))
        return terms[idx]
    
    # here, I redefine datapoint_length; should not be necessary
    # TODO: fix
    if task in ('Decoding', 'Percent', 'Basis_Points', 'Units', 'Orders'):
        datapoint_length = 1
    elif task in ('Addition', 'Ranges'):
        datapoint_length = 2
    
    # Generate pools from which to draw example values
    training_pool, test_pool = generate_pools()
    
    if task in ('Units', 'Context_Units'):
        # essentially reloading obtain_units twice
        # may be better to factor obtain_units out
        #  of extract_data above to prevent duplication
        # TODO
        units_data = obtain_units(units_loc)
        max_idx = len(units_data) - 1
    
    # Generate example values
    if task == 'Units':
        training_data, training_units, training_targets = generation_loop(training_pool,
                                                                          num_training_examples,
                                                                          units=True)
        test_data, test_units, test_targets = generation_loop(test_pool,
                                                              num_test_examples,
                                                              units=True)

        training_data_numpy = np.array(training_targets).squeeze(1)
        test_data_numpy = np.array(test_targets).squeeze(1)
    elif task == 'Context_Units':
        assert(data_loc is not None)
        assert(units_loc is not None)
        sentences, numbers, units = extract_data(data_loc, units_loc)
        # split into train and test sets
        (train_sents, 
         test_sents, 
         train_numbers, 
         test_numbers, 
         train_units,
         test_units) = train_test_split(sentences, numbers, units, test_size=0.2, random_state=123)
        
        # Despite Transformers documentation, produces warning:
        #  Using sep_token, but it is not set yet.
        sep_token = tokenizer.sep_token
  
        training_data = [s.split(' ') + [sep_token] + [n] for s, n in zip(train_sents, train_numbers)]
        test_data = [s.split(' ') + [sep_token] + [n] for s, n in zip(test_sents, test_numbers)]
        
        training_data_numpy = np.array(train_units)
        test_data_numpy = np.array(test_units)
    elif task == 'Ranges':
        range_terms = ('{a}-{b}', '{a} to {b}', 'from {a} to {b}', 'from {a}-{b}')
        training_data, training_range_terms = generation_loop(training_pool, 
                                                              num_training_examples,
                                                              ranges=True)
        test_data, test_range_terms = generation_loop(test_pool, 
                                                      num_test_examples,
                                                      ranges=True)
        
        # Convert to Numpy arrays and generate target values
        training_data_numpy = np.array(training_data)

        test_data_numpy = np.array(test_data)
    elif task == 'Orders': # TODO: review order terms
        order_terms = ('thousand', 'K', 'million', 'mln', 'mn', 'm', 'crore', 'billion', 'bn', 'bln', 'trillion')
        order_dict = {'thousand': math.log(1e3),
                      'K': math.log(1e3),
                      'million': math.log(1e6),
                      'mln': math.log(1e6),
                      'mn': math.log(1e6),
                      'm': math.log(1e6),
                      'crore': math.log(1e7),
                      'billion': math.log(1e9),
                      'bn': math.log(1e9),
                      'bln': math.log(1e9),
                      'trillion': math.log(1e12)}
        training_data, training_order_terms = generation_loop(training_pool,
                                                              num_training_examples,
                                                              orders=True)
        test_data, test_order_terms = generation_loop(test_pool,
                                                      num_test_examples,
                                                      orders=True)
        # Convert to Numpy arrays and generate target values
        training_data_numpy = np.array(training_data).squeeze(-1)
        training_order_numpy = np.array([[order_dict[term]] for term in training_order_terms])
        training_order_numpy = training_order_numpy.squeeze(-1)

        test_data_numpy = np.array(test_data).squeeze(-1)
        test_order_numpy = np.array([[order_dict[term]] for term in test_order_terms])
        test_order_numpy = test_order_numpy.squeeze(-1)
    elif task in ('Percent', 'Basis_Points'): #TODO: determine why this needs to be squeezed but Decoding does not
        training_data = generation_loop(training_pool, num_training_examples)
        test_data = generation_loop(test_pool, num_test_examples)
        
        # Convert to Numpy arrays and generate target values
        training_data_numpy = np.array(training_data).squeeze(-1)

        test_data_numpy = np.array(test_data).squeeze(-1)
    else:
        training_data = generation_loop(training_pool, num_training_examples)
        test_data = generation_loop(test_pool, num_test_examples)
        
        # Convert to Numpy arrays and generate target values
        training_data_numpy = np.array(training_data)

        test_data_numpy = np.array(test_data)
    
    # indices for one_hot must be dtype torch.int64
    # targets with class probabilities must be a floating type
    if task == 'ListMax':
        train_tensor = torch.as_tensor(np.argmax(training_data_numpy, axis=1),
                                       dtype=torch.int64)
        training_targets = one_hot(train_tensor, datapoint_length)
        training_targets = training_targets.to(torch.float32).to(device)

        test_tensor = torch.as_tensor(np.argmax(test_data_numpy, axis=1),
                                      dtype=torch.int64)
        test_targets = one_hot(test_tensor, datapoint_length)
        test_targets = test_targets.to(torch.float32).to(device)
    elif task == 'Decoding':
        train_tensor = torch.as_tensor(training_data_numpy)
        training_targets = train_tensor.to(torch.float32).to(device)
        test_tensor = torch.as_tensor(test_data_numpy)
        test_targets = test_tensor.to(torch.float32).to(device)
    elif task == 'Addition':
        train_tensor = torch.as_tensor(np.sum(training_data_numpy, axis=-1))
        training_targets = train_tensor.to(torch.float32).to(device)
        test_tensor = torch.as_tensor(np.sum(test_data_numpy, axis=-1))
        test_targets = test_tensor.to(torch.float32).to(device)
    elif task in ('Percent', 'Basis_Points'):
        if task == 'Percent':
            decimal_convert = 0.01
        else: # Basis Points
            decimal_convert = 0.0001
        train_tensor = torch.as_tensor(training_data_numpy * decimal_convert)
        training_targets = train_tensor.to(torch.float32).to(device)
        test_tensor = torch.as_tensor(test_data_numpy * decimal_convert)
        test_targets = test_tensor.to(torch.float32).to(device)
    # TODO: confirm if this can be integrated into Decoding task above
    elif task in ('Units', 'Context_Units'):
        train_tensor = torch.as_tensor(training_data_numpy)
        training_targets = one_hot(train_tensor, max_idx)
        training_targets = training_targets.to(torch.float32).to(device)
        test_tensor = torch.as_tensor(test_data_numpy)
        test_targets = one_hot(test_tensor, max_idx)
        test_targets = test_targets.to(torch.float32).to(device)
    elif task == 'Ranges': # TODO: ensure that target data with final dimension (:,2) works
        train_tensor_y1 = torch.as_tensor(training_data_numpy[...,0])
        train_tensor_y2 = torch.as_tensor(training_data_numpy[...,1])
        training_targets_y1 = train_tensor_y1.to(torch.float32).to(device)
        training_targets_y2 = train_tensor_y2.to(torch.float32).to(device)
        test_tensor_y1 = torch.as_tensor(test_data_numpy[...,0])
        test_tensor_y2 = torch.as_tensor(test_data_numpy[...,1])
        test_targets_y1 = test_tensor_y1.to(torch.float32).to(device)
        test_targets_y2 = test_tensor_y2.to(torch.float32).to(device)
    elif task == 'Orders': # TODO: ensure that broadcasting occurs on the correct axis
        train_tensor = torch.as_tensor(np.log(training_data_numpy) + training_order_numpy)
        training_targets = train_tensor.to(torch.float32).to(device)
        test_tensor = torch.as_tensor(np.log(test_data_numpy) + test_order_numpy)
        test_targets = test_tensor.to(torch.float32).to(device)
    else:
        raise ValueError('Task should be one of "ListMax", "Decoding", "Addition", "Percent", "Basis_Points", "Units", "Context_Units", "Ranges", or "Orders"')    
    
    # TODO: implement units support here
    # Convert to string format   
    if task in ('Percent', 'Basis_Points'):
        if task == 'Percent':
            added_term = ' percent'
            added_symbol = '%'
        else: # Basis Points
            added_term = ' basis points'
            added_symbol = ' basis points'
        if use_word_format:
            training_data_strings = [[num2words(n) + added_term for n in line]
                                    for line in training_data]
            test_data_strings = [[num2words(n) + added_term for n in line]
                                 for line in test_data]
        else:
            training_data_strings = [[str(n) + added_symbol for n in line]
                                     for line in training_data]
            test_data_strings = [[str(n) + added_symbol for n in line]
                                 for line in test_data]
    elif task == 'Units':
        if use_word_format:
            train_num_unit_pairs = zip([[num2words(n) for n in line]
                                        for line in training_data], 
                                       training_units)
            test_num_unit_pairs = zip([[num2words(n) for n in line]
                                       for line in test_data],
                                      test_units)
        else:
            train_num_unit_pairs = zip([[str(n) for n in line]
                                        for line in training_data],
                                       training_units)
            test_num_unit_pairs = zip([[str(n) for n in line]
                                       for line in test_data],
                                      test_units)
        # TODO: test and likely remove: this appears duplicated below as 'joined_training_data'
        training_data_strings = [[' '.join(n[0] for n in line)] for line in train_num_unit_pairs]
        test_data_strings = [[' '.join(n[0] for n in line)] for line in test_num_unit_pairs]
    elif task == 'Context_Units':
        # use_word_format inappropriate here due to presence of decimals in task
        training_data_strings = [[str(n) for n in line]
                              for line in training_data]
        test_data_strings = [[str(n) for n in line]
                             for line in test_data]
    elif task == 'Ranges':
        # use_word_format inappropriate here due to possibility of hyphenation
        # TODO: review whether to change this
        training_data_strings = [[n[1].format(a=n[0][0], b=n[0][1])] for n in zip(training_data, training_range_terms)]
        test_data_strings = [[n[1].format(a=n[0][0], b=n[0][1])] for n in zip(test_data, test_range_terms)]
    elif task == 'Orders':
        if use_word_format:
            training_data_strings = [[' '.join((num2words(n[0][0]), n[1]))] for n in zip(training_data, training_order_terms)]
            test_data_strings = [[' '.join((num2words(n[0][0]), n[1]))] for n in zip(test_data, test_order_terms)]
        else:
            training_data_strings = [[' '.join((str(n[0][0]), n[1]))] for n in zip(training_data, training_order_terms)]
            test_data_strings = [[' '.join((str(n[0][0]), n[1]))] for n in zip(test_data, test_order_terms)]
    else:
        if use_word_format:
            training_data_strings = [[num2words(n) for n in line]
                                     for line in training_data]
            test_data_strings = [[num2words(n) for n in line] 
                                 for line in test_data]
        else:
            training_data_strings = [[str(n) for n in line] 
                                     for line in training_data]
            test_data_strings = [[str(n) for n in line]
                                 for line in test_data]
        
    # Tokenize via tokenizer
    # Note: input_data submitted to Dataset needs to be tensors, since .size() must be implemented
    # padding=True since some tokenizers and datasets give varying-length data
    joined_training_data = [' '.join(line) for line in training_data_strings]
    joined_test_data = [' '.join(line) for line in test_data_strings]    
    
    # TODO: is_split_into_words
    if task == 'Context_Units':
        tokenizer.padding_side = 'left'
        padding_pattern = 'max_length'
        max_length = 128
        training_data_tokenized = tokenizer(joined_training_data,
                                            padding=padding_pattern,
                                            max_length=max_length,
                                            return_tensors="pt").to(device)
        test_data_tokenized = tokenizer(joined_training_data,
                                        padding=padding_pattern,
                                        max_length=max_length,
                                        return_tensors="pt").to(device)
    else:
        training_data_tokenized = tokenizer(joined_training_data,
                                            padding=True,
                                            return_tensors="pt").to(device)
    
        test_data_tokenized = tokenizer(joined_test_data,
                                        padding=True,
                                        return_tensors="pt").to(device)
    
    # temporary testing only
#     print(training_data_tokenized.input_ids.size())
#     print(test_data_tokenized.input_ids.size())
    
    # decoder_inputs: 0 is the start symbol for the decoder, 1 end of sequence; used as a placeholder
    training_decoder_inputs = torch.tensor([0,1]).repeat(num_training_examples, 
                                                         1)
    training_decoder_inputs = training_decoder_inputs.to(device)
    
    test_decoder_inputs = torch.tensor([0,1]).repeat(num_test_examples, 
                                                     1)
    test_decoder_inputs = test_decoder_inputs.to(device)
    
    # Store in a Dataset object
    if task == 'Ranges':
        training_dataset = RangeProbingDataset(training_data_tokenized,
                                               training_decoder_inputs,
                                               training_targets_y1,
                                               training_targets_y2)
        test_dataset = RangeProbingDataset(test_data_tokenized,
                                           test_decoder_inputs,
                                           test_targets_y1,
                                           test_targets_y2)
    else:
        training_dataset = ProbingDataset(training_data_tokenized, 
                                          training_decoder_inputs, 
                                          training_targets)
        test_dataset = ProbingDataset(test_data_tokenized, 
                                      test_decoder_inputs, 
                                      test_targets)
    
    return training_dataset, test_dataset
