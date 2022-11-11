import numpy as np

import torch

class OnlineCode:
    def __init__(self, chunk_size, num_classes):
        self._t_1 = chunk_size
        self._K = num_classes
        self._log_sum = 0.0
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
        label_indices = torch.argmax(labels, axis=1)[:, None]
        normalized_outputs = torch.nn.Softmax(dim=1)(outputs)
        values = normalized_outputs.gather(1, label_indices)
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
