import numpy as np

class OnlineCode:
    def __init__(self, chunk_size, num_classes):
        self._t_1 = chunk_size
        self._K = num_classes
        self._log_sum = 0.0
        # runs as first step
        self.update_with_uniform_codelength()

    def update_with_uniform_codelength(self):
        # confirm whether this should be _t_1 or n outright
        self._log_sum += self._t_1 * np.log2(self._K)

    def update_with_results(self, outputs, labels):
        prod = np.product(outputs[labels.astype(bool)])
        self._log_sum += (-1 * np.log2(prod))

    def get_prequential_codelength(self):
        return self._t_1 * np.log2(self._K) + self._log_sum

    def get_compression(self, n):
        # confirm expected value for joint just q(x, y) is
        #  the prequential codelength, or something further
        e_q = get_prequential_codelength()
        uniform_encoding = self.n * np.log2(self._K)
        return uniform_encoding - codelength
