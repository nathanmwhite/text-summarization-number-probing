# Text summarization number probing
# Original code Copyright © 2023 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2023 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import re

import numpy as np

STRING_TO_NUM = {'one': 1,
                 'two': 2,
                 'three': 3,
                 'four': 4,
                 'five': 5,
                 'six': 6,
                 'seven': 7,
                 'eight': 8,
                 'nine': 9,
                 'ten': 10,
                 'zero': 0}


# TODO:
# 1) determine whether numbers with currencies or orders of magnitude are being rejected as numbers
# 2) revisit whether ordinal numbers need to be supported
def is_a_number(sequence, include_hyphen=False):
    """
    is_a_number : checks if a character sequence is a number, possibly containing
        a thousands comma or a decimal point, or is a number between one and ten
        as a string. This function allows the hallucination evaluator to find the
        set of numbers in input and output to compare.
    @param sequence (str) : a string containing the sequence to check if a number
    @param include_hyphen (bool) : whether to allow hyphens when recognizing a
         number, such as with negative numbers or ranges.
    returns : boolean representing whether the sequence is a number given the
        definition here.
    """
    if include_hyphen:
        symbols = (',', '-', '.')
    else:
        symbols = (',', '.', '%')
        
    if sequence == '':
        return False
    elif sequence[0].isdigit() and sequence[-1].isdigit():
        for c in sequence[1:-1]:
            if c.isdigit() or c in symbols:
                continue
            else:
                return False
        return True
    elif sequence in STRING_TO_NUM.keys():
        return True
    else:
        return False
      


def get_numbers(text):
    """
    get_numbers : takes a string and returns tokens that are numbers.
    @param text (str) : the string in which to find the number tokens.
    returns : List[str] containing the strings that represent the numbers
        found.
    """
    tokens = text.split(' ')
    numbers = [w for w in tokens if is_a_number(w)]
    numbers = [STRING_TO_NUM[n] if n in STRING_TO_NUM.keys() else n for n in numbers]
    return numbers


def _found_input_only(input_numbers, output_numbers):
    input_only_items = []
    for item in input_numbers:
        if item not in output_numbers:
            input_only_items.append(item)
    return len(input_only_items)


def _found_output_only(input_numbers, output_numbers):
    output_only_items = []
    for item in output_numbers:
        if item not in input_numbers:
            output_only_items.append(item)
    return len(output_only_items)


def _found_shared(input_numbers, output_numbers):
    shared_items = []
    for item in input_numbers:
        if item in output_numbers:
            shared_items.append(item)
    return len(shared_items)

# TODO:
# 1) add support to avoid differences between 'first half of the year' and 'six months'
# Note: it may be best to ignore this issue, as these scenarios cannot exhaustively be
#    considered, though supporting some may be demonstrative for peer review
def check_numerical(input_strings, output_strings):
    """
    check_numerical : checks lists of input and output strings with the same
        number of string elements to determine how many numerical items
        are found in each and shared. This function supports numbers represented
        as digits as well as their text representation from one to ten. However,
        it does not check for the following:
        1) matching pairs where one has a digit representation and the other the
            text representation for the same number; and
        2) matching pairs where one has a representation in one kind of unit and
            the other in another kind of unit, such as "first quarter" versus
            "first three months."
    @param input_strings (List[str]) : the input strings to check
    @param output_strings (List[str]) : the output strings to check
    returns : List[tuple], where each 3-tuple contains the following:
        1) the number of numerical items found only in the input string
        2) the number of numerical items found only in the output string
        3) the number of numerical items shared between the two.
    """
    # debug
    print(len(input_strings), ',', len(output_strings))
    
    assert(len(input_strings) == len(output_strings))
    
    results = []
    
    for i in range(len(input_strings)):
        
        input_numbers = get_numbers(input_strings[i])
        output_numbers = get_numbers(output_strings[i])
        
        a = _found_input_only(input_numbers, output_numbers)
        b = _found_output_only(input_numbers, output_numbers)
        c = _found_shared(input_numbers, output_numbers)
        results.append((a, b, c))
        
    results = np.asarray(results)
    
    return results


def retokenize(text_sequences):
    """
    retokenize : retokenizes model output strings to follow the spacing patterns of the input,
        to ensure recognition of matching digit-based numerical values where they appear.
    @param text_sequences (List[str]): text sequences to retokenize
    returns : List[str] containing the retokenized sequences
    """
    multiplier_abbrevs = ['mn', 'm', 'mln', 'bln', 'bn', 'b', 'k', 'K', 'tn']
    currency_symbols = ['\$', 'USD', 'CAD', 'C\$', 'AUD', 'A\$', 'skr', 'SEK', 'HK\$', 'Rmb', \
                        'RMB', '£', '¥', 'Y', '₩', '₽', 'CHF', '€', 'EUR', 'eur']
    
    out_sequences = []
    for item in text_sequences:
        sequence = item.rstrip('\n')
        sequence = re.sub(', ', ' , ', sequence)
        # separate for abbreviations; the final product only has the original
        #  sequence plus codes
        sequence = re.sub('\. ', ' . ', sequence) # cannot believe I did this! again!
        sequence = re.sub('\.$', ' .', sequence)
        sequence = re.sub(' \(', ' ( ', sequence)
        sequence = re.sub('\) ', ' ) ', sequence)
        sequence = re.sub('%', ' %', sequence)
        sequence = re.sub('([0-9]+)-([0-9]+)', '\g<1> - \g<2>', sequence)
        sequence = re.sub('-([0-9]+)', '- \g<1>', sequence)
        
        # provides support for the messy "$26bn" type sequences
        multiplier_re = '|'.join(multiplier_abbrevs)
        currency_re = '|'.join(currency_symbols)
        sequence = re.sub('(^| )('+currency_re+')([0-9,.\-]+)('+multiplier_re+'|)', '\g<1> \g<2> \g<3> \g<4>', sequence)
        sequence = re.sub('([0-9,.\-]+)('+multiplier_re+'|) ', '\g<1> \g<2> ', sequence)
        sequence = re.sub(' +', ' ', sequence)
        out_sequences.append(sequence)
        
    return out_sequences


def load_malo_data(path):
    """
    load_malo_data : loads a filtered version of the Financial Phrasebank
        dataset of Malo et al. (2014).
    @param path (str) : the path to the preprocessed Malo et al. dataset
    returns : List[str], where each string is one input line from the dataset.
    """
    with open(path, 'r') as f:
        data = f.readlines()
    sequences = retokenize(data)
    
    return data
