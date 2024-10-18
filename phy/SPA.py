from typing import List
import numpy as np

class SPA:
    def __init__(self, H: np.ndarray, max_iterations: int = 1000, trace_on: bool = True) -> None:
        self.H = H
        self.max_iterations = max_iterations
        self.trace_on = trace_on
        self.H_rows, self.H_cols = H.shape
        self.H_mirror = (self.H + np.ones_like(self.H)) % 2

    def _non_return_to_zero(self, values: np.ndarray) -> np.ndarray:
        return np.where(values >= 0, 0, 1)

    def _calculate_E(self, E: np.ndarray, M: np.ndarray) -> np.ndarray:
        M = np.tanh(M / 2) + self.H_mirror
        non_zero_indices = np.nonzero(self.H)
        m_j_prod = np.prod(M, axis=1)

        for j, i in zip(*non_zero_indices):
            if M[j, i] != 0:
                denominator = m_j_prod[j] / M[j, i]
                if np.abs(denominator) < 1:
                    E[j, i] = np.log((1 + denominator) / (1 - denominator))
                else:
                    E[j, i] = 0
            else:
                E[j, i] = 0
        return E

    def _calculate_M(self, M: np.ndarray, E: np.ndarray, r: np.ndarray) -> np.ndarray:
        non_zero_indices = np.nonzero(self.H)
        for j, i in zip(*non_zero_indices):
            M[j, i] = np.sum(E[:, i]) - E[j, i] + r[i]
        return M * self.H

    def decode(self, received_signal: List[complex]) -> List[int]:
        decoded_message = []
        chunk_size = self.H_cols

        for start_idx in range(0, len(received_signal), chunk_size):
            r_chunk = received_signal[start_idx:start_idx + chunk_size]

            if len(r_chunk) < chunk_size:
                r_chunk = np.pad(r_chunk, (0, chunk_size - len(r_chunk)), 'constant')

            M = np.zeros_like(self.H, dtype=float)
            E = np.zeros_like(self.H, dtype=float)
            l = np.zeros_like(r_chunk, dtype=float)

            for iteration in range(self.max_iterations):
                if iteration == 0:
                    M = np.outer(self.H, r_chunk)

                E = self._calculate_E(E, M)
                l = r_chunk + np.sum(E, axis=0)
                l_hard = self._non_return_to_zero(l)

                if np.all(np.dot(self.H, l_hard) % 2 == 0):
                    break

                M = self._calculate_M(M, E, r_chunk)

            decoded_message.extend(l_hard.tolist())

        return decoded_message
