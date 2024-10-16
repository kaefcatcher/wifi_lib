from typing import List
import numpy as np


def bcc_encoder(data: List[int], code_rate: str = "1/2") -> List[int]:
    """
    Perform BCC encoding based on the IEEE 802.11n standard.

    Parameters:
    - data (List[int]): The data bits to be encoded.
    - code_rate (str): The coding rate. Supported values: "1/2", "2/3", "3/4", "5/6".

    Returns:
    - List[int]: The encoded bits after applying the puncturing pattern.
    """

    g1 = 0b1111001
    g0 = 0b1011011
    shift_register = [0] * 7
    raw_encoded_bits = []

    puncturing_patterns = {
        "1/2": [1, 1],
        "2/3": [1, 1, 1, 0],
        "3/4": [1, 1, 1, 0, 0, 1],
        "5/6": [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]
    }
    pattern = puncturing_patterns.get(code_rate, [1, 1])
    pattern_len = len(pattern)

    for bit in data:
        shift_register = [bit] + shift_register[:-1]

        encoded_bit_0 = sum([shift_register[i]
                            for i in range(7) if (g0 >> i) & 1]) % 2
        encoded_bit_1 = sum([shift_register[i]
                            for i in range(7) if (g1 >> i) & 1]) % 2

        raw_encoded_bits.extend([encoded_bit_0, encoded_bit_1])

    encoded_bits = [bit for i, bit in enumerate(
        raw_encoded_bits) if pattern[i % pattern_len] == 1]

    return encoded_bits


def bcc_decoder(encoded_data: List[int], original_data_len: int,
                code_rate: str = "1/2", constraint_len: int = 7) -> List[int]:
    """"
    BCC decoder (Viterbi Algorithm) according to 802.11n.

    Parameters:
    - encoded_data (List[int]): initial encoded data bits.
    - original_data_len: original data bits length.
    - code_rate (str): code_rate at which encoded_bits should be evaluated.
    - constraint_len: number of shift registers in the scheme (7 by default)

    Returns:
    - decoded_data (List[int]): decoded data of original data length.
    """

    n_states = 2 ** (constraint_len - 1)
    inf = float('inf')

    puncturing_patterns = {
        "1/2": [1, 1],
        "2/3": [1, 1, 1, 0],
        "3/4": [1, 1, 1, 0, 0, 1],
        "5/6": [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]
    }
    pattern = puncturing_patterns.get(code_rate, [1, 1])
    pattern_len = len(pattern)

    path_metrics = [inf] * n_states
    path_metrics[0] = 0
    backtrack = [{} for _ in range(len(encoded_data) // 2)]

    g1 = 0b1111001
    g0 = 0b1011011
    pattern_index = 0

    for i in range(0, len(encoded_data), 2):
        if pattern[pattern_index % pattern_len] == 0:

            pattern_index += 2
            continue

        new_metrics = [inf] * n_states
        current_index = i // 2

        for state in range(n_states):
            if path_metrics[state] < inf:
                for bit in [0, 1]:
                    next_state = ((state << 1) | bit) & 0b111111
                    out_bit_0 = sum(
                        [((state << 1) | bit) >> j & 1 for j in range(7) if (g0 >> j) & 1]) % 2
                    out_bit_1 = sum(
                        [((state << 1) | bit) >> j & 1 for j in range(7) if (g1 >> j) & 1]) % 2

                    branch_metric = (
                        encoded_data[i] != out_bit_0) + (encoded_data[i + 1] != out_bit_1)
                    metric = path_metrics[state] + branch_metric

                    if metric < new_metrics[next_state]:
                        new_metrics[next_state] = metric
                        backtrack[current_index][next_state] = (state, bit)

        path_metrics = new_metrics
        pattern_index += 2

    last_state = np.argmin(path_metrics)
    decoded_data = []

    for i in range(len(encoded_data) // 2 - 1, -1, -1):
        if last_state in backtrack[i]:
            prev_state, bit = backtrack[i][last_state]
            decoded_data.insert(0, bit)
            last_state = prev_state
        else:
            decoded_data.insert(0, 0)

    return decoded_data[:original_data_len]
