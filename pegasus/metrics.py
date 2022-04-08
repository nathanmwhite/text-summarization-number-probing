# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

from sklearn.metrics import accuracy_score
from torch import argmax


def accuracy(y_true, y_pred):
    """
    accuracy : calculates basic accuracy 
    """
    argmax_true = argmax(y_true)
    argmax_pred = argmax(y_pred)
    return accuracy_score(argmax_true, argmax_pred)
