import unittest
import numpy as np
from phy.modulation import bpsk, qpsk, qam16, qam64


class TestModulation(unittest.TestCase):
    def test_bpsk(self):
        # Test BPSK with a simple bit sequence
        bits = [0, 1, 0, 1]
        expected_symbols = [-1 + 0j, 1 + 0j, -1 + 0j, 1 + 0j]
        result = bpsk(bits, constellation=False)
        self.assertEqual(result, expected_symbols)

    def test_qpsk(self):
        # Test QPSK with a simple bit sequence
        bits = [0, 0, 0, 1, 1, 0, 1, 1]
        expected_symbols = [
            (-1 - 1j) / (2**0.5),
            (-1 + 1j) / (2**0.5),
            (1 - 1j) / (2**0.5),
            (1 + 1j) / (2**0.5),
        ]
        result = qpsk(bits, constellation=False)
        self.assertEqual(result, expected_symbols)

    def test_qpsk_invalid_bits(self):
        bits = [0, 1, 0]
        with self.assertRaises(AssertionError):
            qpsk(bits, constellation=False)

    def test_qam16(self):
        # Test 16-QAM with a simple bit sequence
        bits = [0, 0, 0, 0, 1, 1, 1, 1]
        expected_symbols = [(-3 - 3j) / (10**0.5), (1 + 1j) / (10**0.5)]
        result = qam16(bits, constellation=False)
        self.assertEqual(result, expected_symbols)

    def test_qam16_invalid_bits(self):
        # Test that 16-QAM raises assertion error for incorrect number of bits
        bits = [0, 1, 0]
        with self.assertRaises(AssertionError):
            qam16(bits, constellation=False)

    def test_qam64(self):
        # Test 64-QAM with a simple bit sequence
        bits = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        expected_symbols = [(-7 - 7j) / (42**0.5), (3 + 3j) / (42**0.5)]
        result = qam64(bits, constellation=False)

        for res, exp in zip(result, expected_symbols):
            self.assertTrue(
                np.isclose(res, exp, rtol=1e-5, atol=1e-8),
                msg=f"Result {res} is not close to expected {exp}",
            )

    def test_qam64_invalid_bits(self):
        # Test that 64-QAM raises assertion error for incorrect number of bits
        bits = [0, 1, 0]
        with self.assertRaises(AssertionError):
            qam64(bits, constellation=False)


if __name__ == "__main__":
    unittest.main()
