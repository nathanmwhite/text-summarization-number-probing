# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import math
import random

import num2words

import numpy as np

import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

from util import obtain_units

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
                  units_loc: str=None):
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
        ListMax, Decoding, Addition, Percent, Basis_Points, Units
    @param datapoint_length (int) : Number of elements in each datapoint
    @param use_word_format (bool) : Indicates whether to convert an 
        integer into:
            word representation such as 37 --> "thirty-seven" (True), or
            keep as integer representation in a string as in 37 --> "37"
             (False)
    @param units_loc (str) : Indicates location of text file containing
        units as lines, with variants separated by slashes ('/'), where
        'Units' is task
    returns : two ListMaxDatasets: training, test datasets
    """
    def generate_pools():
        # the definition from Wallace et al. could mean:
        # 1. they shuffled and split based on the integer numbers, or
        # 2. they shuffled and split based on string representations (unlikely because Gaussian wouldn't work then)
        # I conclude they shuffled and split based on the integer numbers, meaning different integers would be seen in training and testing
        #  --checked: This is in fact what they did in their code: numeracy/max.py, lines 143-149
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
            return singular_form, target_index
        else:
            return plural_form, target_index
    
    # TODO: evaluate whether to tease out two version for the different tasks
    def generation_loop(pool, num_examples, units=False):
        assembled_data = []
        if units:
            assembled_units = []
            assembled_targets = []
        for i in range(num_examples):
            datapoint = []
            if units:
                datapoint_units = []
                datapoint_targets = []
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
        return assembled_data
 
    if task in ('Decoder', 'Percent', 'Basis_Points', 'Units'):
        datapoint_length = 1
    elif task == 'Addition':
        datapoint_length = 2
    
    # Generate pools from which to draw example values
    training_pool, test_pool = generate_pools()
                                       
    # Generate example values
    if task == 'Units':
        units_data = obtain_units(units_loc)
        training_data, training_units, training_targets = generation_loop(training_pool,
                                                                          num_training_examples,
                                                                          units=True)
        test_data, test_units, test_targets = generation_loop(test_pool,
                                                              num_test_examples,
                                                              units=True)

        training_data_numpy = np.array(training_targets)
        test_data_numpy = np.array(test_targets)
        
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
    elif task == 'Units':
        train_tensor = torch.as_tensor(training_data_numpy)
        training_targets = train_tensor.to(torch.float32).to(device)
        test_tensor = torch.as_tensor(test_data_numpy)
        test_targets = test_tensor.to(torch.float32).to(device)
    else:
        raise ValueError('Task should be one of "ListMax", "Decoding", "Addition", "Percent", "Basis_Points", or "Units"')    
    
    # TODO: implement units support here
    # Convert to string format   
    if task in ('Percent', 'Basis Points'):
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
                                        for n in training_data], 
                                       training_units)
            test_num_unit_pairs = zip([[num2words(n) for n in line]
                                       for n in test_data],
                                      test_units)
        else:
            train_num_unit_pairs = zip([[str(n) for n in line]
                                        for n in training_data],
                                       training_units)
            test_num_unit_pairs = zip([[str(n) for n in line]
                                       for n in test_data],
                                      test_units)
        training_data_strings = [[' '.join(n[0] for n in line)] for line in train_num_unit_pairs]
        test_data_strings = [[' '.join(n[0] for n in line)] for line in test_num_unit_pairs]
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
    joined_training_data = [' '.join(line) for line in training_data_strings]
    training_data_tokenized = tokenizer(joined_training_data,
                                        return_tensors="pt").to(device)
    
    joined_test_data = [' '.join(line) for line in test_data_strings]
    test_data_tokenized = tokenizer(joined_test_data, 
                                    return_tensors="pt").to(device)
    
    # decoder_inputs: 0 is the start symbol for the decoder, 1 end of sequence; used as a placeholder
    training_decoder_inputs = torch.tensor([0,1]).repeat(num_training_examples, 
                                                         1)
    training_decoder_inputs = training_decoder_inputs.to(device)
    
    test_decoder_inputs = torch.tensor([0,1]).repeat(num_test_examples, 
                                                     1)
    test_decoder_inputs = test_decoder_inputs.to(device)
    
    # Store in a Dataset object
    training_dataset = ProbingDataset(training_data_tokenized, 
                                      training_decoder_inputs, 
                                      training_targets)
    test_dataset = ProbingDataset(test_data_tokenized, 
                                  test_decoder_inputs, 
                                  test_targets)
    
    return training_dataset, test_dataset
