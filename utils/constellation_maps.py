import numpy as np

# BPSK Constellation
BPSK_CONSTELLATION = [-1 + 0j, 1 + 0j]

# QPSK Constellation
QPSK_CONSTELLATION = [
    (1 / np.sqrt(2)) * (1 + 1j),   # 00
    (1 / np.sqrt(2)) * (-1 + 1j),  # 01
    (1 / np.sqrt(2)) * (-1 - 1j),  # 10
    (1 / np.sqrt(2)) * (1 - 1j)    # 11
]

# 16-QAM Constellation
QAM16_CONSTELLATION = [
    (1 / np.sqrt(10)) * (-3 + -3j),   # 0000
    (1 / np.sqrt(10)) * (-3 + -1j),   # 0001
    (1 / np.sqrt(10)) * (-3 + 1j),    # 0010
    (1 / np.sqrt(10)) * (-3 + 3j),    # 0011
    (1 / np.sqrt(10)) * (-1 + -3j),   # 0100
    (1 / np.sqrt(10)) * (-1 + -1j),   # 0101
    (1 / np.sqrt(10)) * (-1 + 1j),    # 0110
    (1 / np.sqrt(10)) * (-1 + 3j),    # 0111
    (1 / np.sqrt(10)) * (1 + -3j),     # 1000
    (1 / np.sqrt(10)) * (1 + -1j),     # 1001
    (1 / np.sqrt(10)) * (1 + 1j),      # 1010
    (1 / np.sqrt(10)) * (1 + 3j),      # 1011
    (1 / np.sqrt(10)) * (3 + -3j),     # 1100
    (1 / np.sqrt(10)) * (3 + -1j),     # 1101
    (1 / np.sqrt(10)) * (3 + 1j),      # 1110
    (1 / np.sqrt(10)) * (3 + 3j)       # 1111
]

# 64-QAM Constellation


def generate_64qam_constellation():
    constellation = []
    for i in range(8):
        for j in range(8):
            real = (2 * i - 7)  # Mapping for -7 to 7
            imag = (2 * j - 7)  # Mapping for -7 to 7
            constellation.append((1 / np.sqrt(42)) * (real + 1j * imag))
    return constellation


QAM64_CONSTELLATION = generate_64qam_constellation()
