# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

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

# TODO: Gaussian sampling function
def sample_gaussian(upper_bound):
    pass
