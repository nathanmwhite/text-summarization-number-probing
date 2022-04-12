# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"


def check_arguments(args):
    if args.sample_min_int >= args.sample_max_int:
        return ValueError('Argument sample_min_int is too high compared with sample_max_int.')
    if args.sample_min_float >= args.sample_max_float:
        return ValueError('Argument sample_min_float is too high compared with sample_max_float.')
    if args.float and args.use_words:
        return ValueError('Arguments float and use_words cannot both be True.')
    if args.epochs <= 0:
        return ValueError('Argument epochs has an invalid value. It must be greater than 0.')
    if args.training_examples <= 0:
        return ValueError('Argument training_examples has an invalid value. It must be greater than 0.')
    if args.test_examples <= 0:
        return ValueError('Argument test_examples has an invalid value. It must be greater than 0.')
