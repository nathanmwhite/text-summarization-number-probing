# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import math

import numpy as np

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

# TODO: for List Maximum, convert to strings and enable integer/float/negative/words representations
# TODO: for List Maximum, generate correct answers for training and test sets

sample_max = 99
sample_min = 0
num_train_examples = 100000
num_test_examples = 10000

datapoint_length = 5

def generate_pools(sample_min, sample_max):
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

def sample_gaussian(pool, sample_min, sample_max):
    sample_random_integer = np.random.choice(pool)
    gaussian_sample = np.random.normal(scale=(sample_max - sample_min) * 0.01)
    add_result = sample_random_integer + gaussian_sample
    nearest_value = pool[np.argmin(np.abs(pool - add_result))]
    
    return nearest_value

def generation_loop(pool, sample_min, sample_max, num_examples, datapoint_length):
    assembled_data = []
    for i in range(num_examples):
        datapoint = []
        for j in range(datapoint_length):
            value = sample_gaussian(pool, sample_min, sample_max)
            while value in datapoint:
                value = sample_gaussian(pool, sample_min, sample_max)
            datapoint.append(value)
        assembled_data.append(datapoint)
    
    return assembled_data

# their description does not specify what happens to the obtained value via the Gaussian process
# their code shows that the Gaussian is run five times per data point and appended
# this code follows the description in Wallace et al. (2019); their code has additional complications
#  and behavior not justified by their description
def generate_data(sample_min, sample_max, num_training_examples, num_test_examples, datapoint_length):
    training_pool, test_pool = generate_pools(sample_min, sample_max)
    
    training_data = generation_loop(training_pool, sample_min, sample_max, num_training_examples, datapoint_length)
    
    test_data = generation_loop(test_pool, sample_min, sample_max, num_test_examples, datapoint_length)
    
    return training_data, test_data
        
    
