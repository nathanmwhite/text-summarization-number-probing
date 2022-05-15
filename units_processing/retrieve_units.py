#!/usr/bin/python

# text-summarization-number-probing
# Original code Copyright © 2022 Nathan M. White
"""
This file contains a script to locate and log the most common
 single-word unit terms in a dataset.
This file also contains the useful ancillary function
 is_a_number, which checks text to see if a number is present,
 including hyphens and commas.
"""

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import logging

logging.basicConfig(filename='retrieve_units.log', level=logging.INFO)


def is_a_number(thing):
    """
    is_a_number : a function that checks if a string contains
        a number, including hyphens and commas that normally
        appear in a number in text
    @param thing (str) : string to check if it is a number
    returns bool : True if the string represents a numerical
        value, False if it does not
    """
    if thing == '':
        return False
    if thing[0].isdigit() and thing[-1].isdigit():
        for c in thing[1:-1]:
            if c.isdigit() or c in (',', '-'):
                continue
            else:
                return False
    else:
        return False
    return True


# TODO: consider also searching for sequences ending in -s,
#   or using stopwords to split
#   this currently only considers single-word units
if __name__ == '__main__':
    items_found = []
    for element in data_list:
        matched_thing = []
        get_next = False
        for word in element.split(' '):
            if is_a_number(word):
                get_next = True
                matched_thing.append(word)
            elif get_next == True:
                if word in ('million', 'mn', 'm', 'mln', 'bln', 'bn', 
                            'b', 'billion', 'square', 'sq.'):
                    matched_thing.append(word)
                else:
                    get_next = False
                    matched_thing.append(word)
                    items_found.append(matched_thing)
                    matched_thing = []
                    
    fd = FreqDist(w[-1] for w in items_found 
                  if w[-1] not in string.punctuation 
                  and w[-1] not in eng)
    logging.log(fd.most_common(500))
