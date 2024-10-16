from typing import List
import numpy as np
import unittest
from utils.LDPC_matrix import LDPCMatrixGenerator

def rref(A: np.ndarray) -> np.ndarray:
    """Compute the Reduced Row Echelon Form (RREF) of a matrix."""
    A = A.copy().astype(float)
    rows, cols = A.shape
    pivot_row = 0

    for pivot_col in range(cols):
        if pivot_row >= rows:
            break

        max_row = np.argmax(np.abs(A[pivot_row:rows, pivot_col])) + pivot_row
        if A[max_row, pivot_col] == 0:
            continue

        A[[pivot_row, max_row]] = A[[max_row, pivot_row]]

        A[pivot_row] = A[pivot_row] / A[pivot_row, pivot_col]

        for r in range(rows):
            if r != pivot_row:
                A[r] -= A[r, pivot_col] * A[pivot_row]

        pivot_row += 1

    return A

def create_generator_matrix(ldpc_matrix: np.ndarray) -> np.ndarray:
    """Create a generator matrix based on the LDPC matrix."""
    m, n = ldpc_matrix.shape
    k = n - m

    H_rref = rref(ldpc_matrix)

    P = H_rref[:, :k]

    I_k = np.eye(k, dtype=int)
    G = np.hstack((I_k, P.T))

    return np.array(G)

def encode_with_ldpc(generator_matrix, input_data)->List[int]:
    """Encode the input data using the LDPC generator matrix."""
    k, n = generator_matrix.shape
    m = n - k

    if len(input_data) % k != 0:
        padding = k - (len(input_data) % k)
        input_data = np.pad(input_data, (0, padding), 'constant')

    data_blocks = input_data.reshape(-1, k)
    encoded_blocks = []

    for block in data_blocks:
        parity_bits = np.zeros(m, dtype=int)
        for row in generator_matrix:
            info_part = row[:k]
            parity_part = row[k:]
            xor_result = np.bitwise_xor.reduce(block[info_part == 1], initial=0)
            parity_bits = np.bitwise_xor(parity_bits, parity_part.astype(int) * xor_result)

        encoded_block = np.concatenate((block, parity_bits))
        encoded_blocks.append(encoded_block)

    encoded_codeword = np.concatenate(encoded_blocks)
    return encoded_codeword.tolist()

def recover_information_bits(decoded_codeword, k, n)->List[int]:
    """
    Recover the information bits from the decoded codeword.

    Parameters:
    decoded_codeword : numpy array
        The decoded codeword from the SPA decoder
    k : int
        Number of information bits per block
    n : int
        Total number of bits per codeword block (information + parity)

    Returns:
    numpy array : The recovered information bits
    """

    num_blocks = len(decoded_codeword) // n
    reshaped_codeword = decoded_codeword.reshape(num_blocks, n)

    information_bits = reshaped_codeword[:, :k].flatten()

    return information_bits.tolist()
