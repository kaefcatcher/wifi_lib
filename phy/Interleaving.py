from typing import List, Union
import numpy as np


class Interleaving:
    @staticmethod
    def interleaver(data: List[int], N_bpcs: int) -> List[int]:
        """
        Interleaver function based on 802.11n.

        Parameters:
        - data (List[Union[int, complex]]): input data to be interleaved.
        - N_bpcs (int): number of coded bits per subcarrier.

        Returns:
        - List[Union[int, complex]]: interleaved data.
        """
        N_cbps = len(data)
        num_columns = N_cbps // N_bpcs
        interleaved_data: List[int] = [0] * N_cbps

        for k in range(N_cbps):
            row = k % num_columns
            column = k // num_columns

            new_index = row * N_bpcs + column
            interleaved_data[new_index] = data[k]

        return interleaved_data

    @staticmethod
    def deinterleaver(interleaved_data: List[int], N_bpcs: int) -> List[int]:
        """
        Deinterleaver function based on 802.11n.

        Parameters:
        - interleaved_data (List[Union[int, complex]]): input data to be deinterleaved.
        - N_bpcs (int): number of coded bits per subcarrier.

        Returns:
        - List[Union[int, complex]]: deinterleaved data.
        """
        N_cbps = len(interleaved_data)
        num_columns = N_cbps // N_bpcs
        deinterleaved_data: List[int] = [0] * N_cbps

        for k in range(N_cbps):
            row = k % N_bpcs
            column = k // N_bpcs

            original_index = row * num_columns + column
            deinterleaved_data[original_index] = interleaved_data[k]

        return deinterleaved_data
