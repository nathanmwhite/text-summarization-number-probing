# Text summarization number probing
# Original code Copyright © 2023 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2023 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

def is_a_number(thing, include_hyphen=False):
    if include_hyphen:
        symbols = (',', '-', '.')
    else:
        symbols = (',', '.')
        
    if thing == '':
        return False
    elif thing[0].isdigit() and thing[-1].isdigit():
        for c in thing[1:-1]:
            if c.isdigit() or c in symbols:
                continue
            else:
                return False
        return True
    elif thing in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']:
        return True
    else:
        return False
      
      
