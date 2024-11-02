import unittest
from typing import List
from phy.Mapping import Mapping # Replace 'mapping' with the actual module name if different
from utils.constellation_maps import (
    BPSK_CONSTELLATION,
    QPSK_CONSTELLATION,
    QAM16_CONSTELLATION,
    QAM64_CONSTELLATION
)

class TestMapping(unittest.TestCase):
    def setUp(self):
        # Example input data for each modulation scheme
        self.bpsk_bits: List[int] = [0, 1, 1, 0]
        self.qpsk_bits: List[int] = [0, 1, 1, 0, 1, 0]
        self.qam16_bits: List[int] = [0, 1, 1, 0, 1, 0, 0, 1]
        self.qam64_bits: List[int] = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]

    def test_bpsk_mapping(self):
        # Test BPSK modulation
        symbols = Mapping.bpsk(self.bpsk_bits, constellation=False)
        expected_symbols = [BPSK_CONSTELLATION[bit] for bit in self.bpsk_bits]
        self.assertEqual(symbols, expected_symbols, "BPSK mapping did not produce expected symbols")

    def test_qpsk_mapping(self):
        # Test QPSK modulation
        symbols = Mapping.qpsk(self.qpsk_bits, constellation=False)
        expected_symbols = [QPSK_CONSTELLATION[self.qpsk_bits[i] * 2 + self.qpsk_bits[i + 1]] for i in range(0, len(self.qpsk_bits), 2)]
        self.assertEqual(symbols, expected_symbols, "QPSK mapping did not produce expected symbols")

    def test_qam16_mapping(self):
        # Test 16-QAM modulation
        symbols = Mapping.qam16(self.qam16_bits, constellation=False)
        expected_symbols = [
            QAM16_CONSTELLATION[(self.qam16_bits[i] << 3) | (self.qam16_bits[i + 1] << 2) | (self.qam16_bits[i + 2] << 1) | self.qam16_bits[i + 3]]
            for i in range(0, len(self.qam16_bits), 4)
        ]
        self.assertEqual(symbols, expected_symbols, "16-QAM mapping did not produce expected symbols")

    def test_qam64_mapping(self):
        # Test 64-QAM modulation
        symbols = Mapping.qam64(self.qam64_bits, constellation=False)
        expected_symbols = [
            QAM64_CONSTELLATION[(self.qam64_bits[i] << 5) | (self.qam64_bits[i + 1] << 4) | (self.qam64_bits[i + 2] << 3) |
                                (self.qam64_bits[i + 3] << 2) | (self.qam64_bits[i + 4] << 1) | self.qam64_bits[i + 5]]
            for i in range(0, len(self.qam64_bits), 6)
        ]
        self.assertEqual(symbols, expected_symbols, "64-QAM mapping did not produce expected symbols")

    def test_hard_decision(self):
        # Test hard decision demapping
        symbols = Mapping.qpsk(self.qpsk_bits, constellation=False)
        demapped_bits = Mapping.hard_decision(symbols, QPSK_CONSTELLATION, 2)
        self.assertEqual(demapped_bits, self.qpsk_bits, "Hard decision demapping did not reproduce the original bits")


if __name__ == "__main__":
    unittest.main()
