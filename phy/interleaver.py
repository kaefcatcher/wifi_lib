from typing import List
import numpy as np


def interleaver(data: List[int], N_bpcs: int) -> List[int]:
    """
    Interleaver function based on the provided MATLAB code.

    Parameters:
    - data (List[int]): input data to be interleaved.
    - N_bpcs (int): number of coded bits per subcarrier.

    Returns:
    - List[int]: interleaved data.
    """
    N_cbps = len(data)
    num_columns = N_cbps // N_bpcs  # Number of columns based on N_bpcs
    interleaved_data = [0] * N_cbps

    for k in range(N_cbps):
        # Calculate the row and column for the current bit
        row = k % num_columns
        column = k // num_columns

        # Calculate the new position in the interleaved data
        new_index = row * N_bpcs + column
        interleaved_data[new_index] = data[k]

    return interleaved_data


def deinterleaver(interleaved_data: List[int], N_bpcs: int) -> List[int]:
    """
    Deinterleaver function based on the provided MATLAB code.

    Parameters:
    - interleaved_data (List[int]): input data to be deinterleaved.
    - N_bpcs (int): number of coded bits per subcarrier.

    Returns:
    - List[int]: deinterleaved data.
    """
    N_cbps = len(interleaved_data)
    num_columns = N_cbps // N_bpcs  # Number of columns based on N_bpcs
    deinterleaved_data = [0] * N_cbps

    for k in range(N_cbps):
        # Calculate the row and column for the current bit
        row = k % N_bpcs
        column = k // N_bpcs

        # Calculate the original position in the deinterleaved data
        original_index = row * num_columns + column
        deinterleaved_data[original_index] = interleaved_data[k]

    return deinterleaved_data
