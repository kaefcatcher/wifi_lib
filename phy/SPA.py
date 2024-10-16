from typing import List
import numpy as np

class SPA:
    def __init__(self, H, Imax=1000, trace_on=True):
        self.H = H
        self.Imax = Imax
        self.trace_on = trace_on
        self.H_0 = np.shape(H)[0]
        self.H_1 = np.shape(H)[1]
        self.H_mirr = (self.H + np.ones(np.shape(self.H))) % 2

    def __nrz(self, l):
        for idx, l_j in enumerate(l):
            if l_j >= 0:
                l[idx] = 0
            else:
                l[idx] = 1
        return l

    def __calc_E(self, E, M):
        M = np.tanh(M / 2) + self.H_mirr
        for j in range(self.H_0):
            for i in range(self.H_1):
                if self.H[j, i] != 0:
                    m_j_prod = np.prod(M[j, :])
                    if M[j, i] != 0:
                        denominator = m_j_prod / M[j, i]
                        if np.abs(denominator) < 1:
                            E[j, i] = np.log((1 + denominator) / (1 - denominator))
                        else:
                            E[j, i] = 0
                    else:
                        E[j, i] = 0
        return E

    def __calc_M(self, M, E, r):
        for j in range(self.H_0):
            for i in range(self.H_1):
                if self.H[j, i] != 0:
                    M[j, i] = np.sum(E[:, i]) - E[j, i] + r[i]
        M = M * self.H
        return M

    def decode(self, r: List[complex]) -> List[int]:
        # Initialize the final output list
        decoded_message = []


        chunk_size = self.H_1
        for start_idx in range(0, len(r), chunk_size):

            r_chunk = r[start_idx:start_idx + chunk_size]

            if len(r_chunk) < chunk_size:
                r_chunk = np.pad(r_chunk, (0, chunk_size - len(r_chunk)), 'constant')

            stop = False
            I = 0
            M = np.zeros(np.shape(self.H))
            E = np.zeros(np.shape(self.H))
            l = np.zeros(np.shape(r_chunk))

            while not stop and I != self.Imax:
                if I == 0:
                    for j in range(self.H_0):
                        M[j, :] = r_chunk * self.H[j, :]
                E = self.__calc_E(E, M)
                l = r_chunk + np.sum(E, axis=0)
                l = self.__nrz(l)
                s = np.dot(self.H, l) % 2
                if np.all(s == 0):
                    stop = True
                else:
                    I += 1
                    M = self.__calc_M(M, E, r_chunk)

            decoded_message.extend(l.tolist())

        return decoded_message
