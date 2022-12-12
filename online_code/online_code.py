import numpy as np

import torch

# TODO: redefine self._K for rmse mode
# TODO: implement proper logic for 'log_rmse' with orders, and
#    'rmse' with ranges, percents, and basis points
# TODO: accuracy approach needs to be able to handle 194 classes
#  as it is, this is subject to the exploding gradient problem (note Raiyan's comments, which are correct)
#  potential solutions:
#   1. apply transformation to results to emphasize correct output; only works if correct label most of the time (would have expected so)
#   2. if not, then test whether longer training cycle would improve results
#   3. could design and implement a gradient clipping mechanism
#       Raiyan's comment: should add instead of multiply; my response: taking the product is the standard case for OnlineCode
#   4. could normalize by the inverse of the number of classes, given that the number of classes influences overall metric values
#  Note similar issue for Corander and Marttinen 2006:
#   "Notice that, under typical improper reference priors, dimension d_j would be too extensive with respect to n,
#    the corresponding posterior expectation of (7) would tend to minus infinity, thus automatically preventing the use of models
#    for which there is not enough data to estimate the parameters."
#  --> This suggests that radically ramping up the number of training set datapoints may prove beneficial, but only if num_classes
#   is still held to a very low value.
# TODO: test RMSE approaches
class OnlineCode:
    def __init__(self, chunk_size, num_classes, mode='acc', x_min=None, x_max=None):
        self._t_1 = chunk_size
        self._K = num_classes
        self._log_sum = 0.0
        if mode not in ['acc', 'rmse', 'log_rmse']:
            raise ValueError("Specified mode for OnlineCode must be one of ['acc', 'rmse', 'log_rmse']")
        self._mode = mode
        if self._mode in ['rmse', 'log_rmse']:
            if (x_min == None or x_max == None) and (type(x_min) not in [int, float] or type(x_max) not in [int, float]):
                raise TypeError("When mode is 'rmse' or 'log_rmse', x_min and x_max must be type int or float")
            if self._mode == 'rmse':
                self._r = x_max - x_min
#             elif self.mode == 'log_rmse':
#                 self._r = np.log(x_max) - np.log(x_min)
        # runs as first step
        self.update_with_uniform_codelength()

    def update_with_uniform_codelength(self):
        self._log_sum += self._t_1 * np.log2(self._K)
        print(f'uniform_codelength calculation: {self._log_sum}')

    def update_with_results(self, outputs, labels):
        """
        update_with_results takes the labels and outputs as inputs and
            finds the product of the probability distribution values
            of each output, selects out the output value corresponding
            to the correct label, takes the product of these output values,
            and takes the log_2 of the product and applies it to the log_sum
        @param outputs (Tensor) : tensor containing the output distribution values
        @param labels (Tensor) : tensor containing one-hot values for the correct
            label for each data point
        """
        # TODO: review approach here
        # TODO: explore how to support non-probability dist values
        # currently supports only probability distributions
        #  so applicable to units
        if self._mode == 'acc':
            label_indices = torch.argmax(labels, axis=1)[:, None]
            normalized_outputs = torch.nn.Softmax(dim=1)(outputs)
            values = normalized_outputs.gather(1, label_indices)
        elif self._mode == 'rmse':
            errors = torch.abs(outputs - labels)
            values = 1 - errors / self._r
        elif self._mode == 'log_rmse':
            raise NotImplementedError('Updates with log_rmse approach are not yet supported.')
        product = torch.prod(values)
        print('Values:', values)
        print('Values average: {n}'.format(n=torch.mean(values)))
        print('Values product: {n}'.format(n=product))
        print('Values product log2: {n}'.format(n=torch.log2(product)))
        self._log_sum += (-1 * torch.log2(product))

    def get_prequential_codelength(self):
        return self._t_1 * np.log2(self._K) + self._log_sum

    def get_compression(self, n):
        # confirm expected value for joint just q(x, y) is
        #  the prequential codelength, or something further
        e_q = self.get_prequential_codelength()
        # TODO: Blier and Ollivier (2018: 6) on Online Code:
        #  "a model with default values is used to encode the first few data"
        #  "such a model may learn from the first k data samples"
        #  "a uniform encoding is used for the first few points"
        #  This suggests that a model with default values (assumed here, untrained model)
        #   is the "uniform encoding"
        #  This currently uses a naive calculation roughly based on Voita and Titov, p. 3:
        #   "Compression is usually compared against uniform encoding which does not require
        #     any learning from data. It assumes p(y|x) = p_unif(y|x) = 1/K, and yields
        #     codelength L_unif(y_1:n|x_1:n) = n log_2 K bits"
        uniform_encoding = n * np.log2(self._K)
        return uniform_encoding - e_q
