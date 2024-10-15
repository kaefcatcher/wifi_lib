from typing import List


def scrambler(seed: int, data_bits: List[int]) -> List[int]:
    """
    Scramble the input data using a 7-bit LFSR scrambler based on IEEE 802.11.2020.

    Parameters:
    - seed (int): The initial seed for the scrambler (7 bits, 0-127).
    - data_bits (List[int]): The input data bits to be scrambled (list of 0s and 1s).

    Returns:
    - List[int]: The scrambled data bits (list of 0s and 1s).
    """
    assert (seed >= 0 and seed <= 127), "Seed must be a 7-bit integer (0-127)."

    lfsr = [(seed >> i) & 1 for i in range(6, -1, -1)]

    scrambled_bits = []

    for bit in data_bits:
        new_bit = lfsr[3] ^ lfsr[6]
        scrambled_bit = bit ^ new_bit
        scrambled_bits.append(scrambled_bit)
        lfsr = [new_bit] + lfsr[:-1]

    return scrambled_bits


def descrambler(seed: int, scrambled_bits: List[int]) -> List[int]:
    """
    Descramble the input data using a 7-bit LFSR descrambler based on IEEE 802.11.2020.

    Parameters:
    - seed (int): The initial seed for the descrambler (7 bits, 0-127).
    - scrambled_bits (List[int]): The scrambled data bits to be descrambled (list of 0s and 1s).

    Returns:
    - List[int]: The descrambled data bits (list of 0s and 1s).
    """

    assert (seed >= 0 and seed <= 127), "Seed must be a 7-bit integer (0-127)."

    lfsr = [(seed >> i) & 1 for i in range(6, -1, -1)]

    descrambled_bits = []

    for bit in scrambled_bits:
        new_bit = lfsr[3] ^ lfsr[6]
        descrambled_bit = bit ^ new_bit
        descrambled_bits.append(descrambled_bit)
        lfsr = [new_bit] + lfsr[:-1]

    return descrambled_bits
