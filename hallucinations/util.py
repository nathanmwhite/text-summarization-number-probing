# Text summarization number probing
# Original code Copyright © 2023 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2023 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

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
        definition here
    """
    if include_hyphen:
        symbols = (',', '-', '.')
    else:
        symbols = (',', '.')
        
    if sequence == '':
        return False
    elif sequence[0].isdigit() and sequence[-1].isdigit():
        for c in sequence[1:-1]:
            if c.isdigit() or c in symbols:
                continue
            else:
                return False
        return True
    elif sequence in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']:
        return True
    else:
        return False
      
      
