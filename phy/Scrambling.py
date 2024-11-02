from typing import List

class Scrambling:
    def __init__(self, seed: int):
        """
        Initialize the Scrambler with a seed.

        Parameters:
        - seed (int): The initial seed for the scrambler (7 bits, 0-127).
        """
        assert (seed >= 0 and seed <= 127), "Seed must be a 7-bit integer (0-127)."
        self.seed = seed
        self._initialize_lfsr()

    def _initialize_lfsr(self):
        """Set the LFSR to the initial state based on the seed."""
        self.lfsr = [(self.seed >> i) & 1 for i in range(6, -1, -1)]

    def scramble(self, data_bits: List[int]) -> List[int]:
        """
        Scramble the input data using a 7-bit LFSR scrambler based on IEEE 802.11.2020.

        Parameters:
        - data_bits (List[int]): The input data bits to be scrambled (list of 0s and 1s).

        Returns:
        - List[int]: The scrambled data bits (list of 0s and 1s).
        """
        self._initialize_lfsr()  # Reset LFSR to the initial state
        scrambled_bits = []

        for bit in data_bits:
            new_bit = self.lfsr[3] ^ self.lfsr[6]
            scrambled_bit = int(bit) ^ new_bit
            scrambled_bits.append(scrambled_bit)
            self.lfsr = [new_bit] + self.lfsr[:-1]

        return scrambled_bits

    def descramble(self, scrambled_bits: List[int]) -> List[int]:
        """
        Descramble the input data using a 7-bit LFSR descrambler based on IEEE 802.11.2020.

        Parameters:
        - scrambled_bits (List[int]): The scrambled data bits to be descrambled (list of 0s and 1s).

        Returns:
        - List[int]: The descrambled data bits (list of 0s and 1s).
        """
        self._initialize_lfsr()  # Reset LFSR to the initial state
        descrambled_bits = []

        for bit in scrambled_bits:
            new_bit = self.lfsr[3] ^ self.lfsr[6]
            descrambled_bit = int(bit) ^ new_bit
            descrambled_bits.append(descrambled_bit)
            self.lfsr = [new_bit] + self.lfsr[:-1]

        return descrambled_bits
