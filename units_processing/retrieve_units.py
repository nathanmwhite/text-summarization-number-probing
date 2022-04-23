# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import logging

logging.basicConfig(filename='retrieve_units.log', level=logging.INFO)


def is_a_number(thing):
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


# TODO: redo logic also to find sequences ending in -s,
#   or using stopwords to split
#   as of now, this does not successfully find unit types
#   that are multiword
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
