# Text summarization number probing
# Original code Copyright © 2023 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2023 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import numpy as np

from ..pegasus.generate_data import sample_float
from ..units_processing.retrieve_units import is_a_number


# get new numbers from continuous uniform space
def generate_number(min=0, max=99, use_floats=False):
    integer = np.random.randint(max)
    decimal = sample_float(integer)
    return decimal


# need to generate numbers in relationships
# first relationship type (start, increase_amount, end) # and decrease!
# second relationship type (start, increase_percent, end)
# third relationship type (start, increase_basis_points, end)
# fourth relationship type (currency 1, currency 2)
# fifth relationship type (numeral, units)
# other relationships need to be considered:
#  base relationships on tasks
#  Addition
#  Ranges
#  Percents
#  Basis Points
#  Orders
#  Units
# Malo examples:
# "In the third quarter of 2010 , net sales increased by 5.2 % to EUR 205.5 mn , and operating profit by 34.9 % to EUR 23.5 mn ."
# "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007 representing 7.7 % of net sales ."
# "Clothing retail chain Sepp+ñl+ñ 's sales increased by 8 % to EUR 155.2 mn , and operating profit rose to EUR 31.1 mn from EUR 17.1 mn in 2004 ."
# "Its board of directors will propose a dividend of EUR0 .12 per share for 2010 , up from the EUR0 .08 per share paid in 2009 ."
# "Net sales surged by 18.5 % to EUR167 .8 m. Teleste said that EUR20 .4 m , or 12.2 % , of the sales came from the acquisitions made in 2009 ."
# "A Helsinki : ELIiV today reported EPS of EUR1 .13 for 2009 , an increase over EPS of EUR1 .12 in 2008 ."
# "Commission income increased by 22 % to EUR 4.4 mn , and lending volume rose by 13.5 % ."
# "The company 's order book stood at 1.5 bln euro $ 2.2 bln on September 30 , 2007 , up by 24.2 pct on the year , with international orders amounting to 365 mln euro $ 534.3 mln ."
# "Seppala 's revenue increased by 0.2 % to EUR10 .1 m. In Finland , revenue went down by 2.4 % to EUR6 .8 m , while sales abroad rose by 6.2 % to EUR3 .3 m. Sales increased in all the Baltic countries as well as in Russia and Ukraine ."
# "In its financial report , published on Friday , SEB said its net profit soared to SEK6 .745 bn in 2010 from a year-earlier SEK1 .114 bn and proposed a 50 % dividend increase to SEK1 .50 per share ."
# "Rinkuskiai raised the sales by 18.1 percent , to 1.37 million liters , while the sales of Kauno Alus grew by 14.3 percent , to 960,000 liters ."


# TODO: determine whether `task` should be a list
def generate_augmented_examples(task, num_examples=int(1e05)):
    """
    generate_augmented_examples creates additional training examples in order to
        test whether additional training examples with better representation
        of numerals results in improved performance on hallucination issues.
    @param task (str) : The task for which improved training is sought.
    @param num_examples (int) : The number of augmented examples to create.
    returns List : augmented examples to add to dataset for model fine-tuning
    """
    amount_increase_examples = [
        "In the third quarter, net sales increased by {amt} from {start} to {end}.",
        "In the first half of the year, operating profit rose by {amt} from {start} to {end}.",
        "Operating profit rose to {end} from {start} in the corresponding period from the prior year, representing an increase of {amt}.",
        "The retail chain's sales increased by {amt} from {start} to {end}.",
        "Net sales surged by {amt} from {start} to {end}.",
        "Commission income increased by {amt} to {end}, up from {start} a year earlier.",
        "The company's revenue increased by {amt} to {end} from {start}.,
        "The company's order book stood at {end}, up by {amt} on the year from {start}.",
        "In its financial report, it said its net profit soared by {amt} to {end} from a year-earlier {start}.",
        "They raised sales by {amt}, to {end}, from {start} a year earlier."
    ]
    amount_decrease_examples = [
        "In the third quarter, net sales decreased by {amt} from {start} to {end}.",
        "In the first half of the year, operating profit fell by {amt} from {start} to {end}.",
        "Operating profit fell to {end} from {start} in the corresponding period from the prior year, representing a decrease of {amt}.",
        "The retail chain's sales decreased by {amt} from {start} to {end}.",
        "Net sales declined by {amt} from {start} to {end}.",
        "Commission income decreased by {amt} to {end}, down from {start} a year earlier.",
        "The company's revenue decreased by {amt} to {end} from {start}.,
        "The company's order book stood at {end}, down by {amt} on the year from {start}.",
        "In its financial report, it said its net profit plummeted by {amt} to {end} from a year-earlier {start}.",
        "They lowered sales by {amt}, to {end}, from {start} a year earlier."
    ]
    # need a source to create models from for basis points: Turenne?
    bp_increase_examples = [
    ]
    # TODO: implement remainder
    return None
