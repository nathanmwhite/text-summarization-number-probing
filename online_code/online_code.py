import numpy as np

class OnlineCode:
    def __init__(self, chunk_size, num_classes):
        self._t_1 = chunk_size
        self._K = num_classes
        self._log_sum = 0.0
        # runs as first step
        self.update_with_uniform_codelength()

    def update_with_uniform_codelength(self):
        self._log_sum += self._t_1 * np.log2(self._K)

    def update_with_results(self, outputs, labels):
        # TODO: fix next line
        # np.prod, not np.product
        # TODO: review approach here
        prod = np.product(outputs[labels.astype(bool)])
        self._log_sum += (-1 * np.log2(prod))

    def get_prequential_codelength(self):
        return self._t_1 * np.log2(self._K) + self._log_sum

    def get_compression(self, n):
        # confirm expected value for joint just q(x, y) is
        #  the prequential codelength, or something further
        e_q = get_prequential_codelength()
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
        return uniform_encoding - codelength
