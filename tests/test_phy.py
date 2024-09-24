import unittest
import numpy as np
from phy.modulation import (
    bpsk, qpsk, qam16, qam64, ofdm_symbol,
    subcarrier_mapping, calculate_subcarrier_frequency_spacing, w_tsym,
)
from phy.scrambler import (
    scrambler, descrambler
)


class TestModulation(unittest.TestCase):
    def test_bpsk(self):
        # Test BPSK modulation with a simple bit sequence
        bits = [0, 1, 0, 1]
        expected_symbols = [-1 + 0j, 1 + 0j, -1 + 0j, 1 + 0j]
        result = bpsk(bits, constellation=False)
        self.assertEqual(result, expected_symbols)

    def test_qpsk(self):
        # Test QPSK modulation with a simple bit sequence
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
        # Test that an error is raised for an odd number of bits in QPSK
        bits = [0, 1, 0]
        with self.assertRaises(AssertionError):
            qpsk(bits, constellation=False)

    def test_qam16(self):
        # Test 16-QAM modulation with a simple bit sequence
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
        # Test 64-QAM modulation with a simple bit sequence
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


class TestOFDM(unittest.TestCase):
    def test_ofdm_symbol(self):
        # Test OFDM symbol generation with valid parameters
        d_k_n = [1] * 48  # Example data symbols
        p_k = [1] * 52    # Example pilot symbols
        channel_bw = 20   # Example channel bandwidth
        t = 0.1           # Example time
        delta_f = calculate_subcarrier_frequency_spacing(channel_bw)

        result = ofdm_symbol(
            d_k_n,
            p_k,
            channel_bw,
            t,
            subcarrier_mapping,
            lambda bw: delta_f,
            lambda t: 1)  # Assuming a rectangular window function

        self.assertIsInstance(result, complex)

    def test_ofdm_invalid_data_length(self):
        # Test OFDM symbol generation with invalid data length
        d_k_n = [1] * 50  # Invalid length
        p_k = [1] * 52
        channel_bw = 20
        t = 0.1

        with self.assertRaises(AssertionError):
            ofdm_symbol(d_k_n, p_k, channel_bw, t,
                        subcarrier_mapping,
                        calculate_subcarrier_frequency_spacing,
                        lambda t: 1)

    def test_ofdm_invalid_channel_bw(self):
        # Test OFDM symbol generation with invalid channel bandwidth
        d_k_n = [1] * 48
        p_k = [1] * 52
        channel_bw = 25  # Invalid bandwidth
        t = 0.1

        with self.assertRaises(AssertionError):
            ofdm_symbol(d_k_n, p_k, channel_bw, t,
                        subcarrier_mapping,
                        calculate_subcarrier_frequency_spacing,
                        lambda t: 1)

    def test_subcarrier_mapping(self):
        # Test subcarrier mapping function with valid input
        self.assertEqual(subcarrier_mapping(0), -26)
        self.assertEqual(subcarrier_mapping(47), 26)

    def test_subcarrier_mapping_invalid_input(self):
        # Test subcarrier mapping with invalid input
        with self.assertRaises(AssertionError):
            subcarrier_mapping(50)

    def test_calculate_subcarrier_frequency_spacing(self):
        # Test subcarrier frequency spacing calculation
        self.assertEqual(calculate_subcarrier_frequency_spacing(20), 0.3125e6)
        self.assertEqual(calculate_subcarrier_frequency_spacing(10), 0.156e6)

    def test_w_tsym(self):
        # Test windowing function calculation
        T_GI = 0.8e-6
        T_FFT = 3.2e-6
        t = 0.1e-6
        result = w_tsym(t, T_GI, T_FFT)
        self.assertIsInstance(result, float)
        self.assertTrue(0 <= result <= 1)

    def test_scrambler_deterministic(self):
        """Test if scrambler is deterministic (same input produces same output)."""
        data_bits = [1, 0, 1, 1, 0, 0, 1, 0]
        seed = 93
        scrambled_once = scrambler(seed, data_bits)
        scrambled_twice = scrambler(seed, data_bits)
        self.assertEqual(
            scrambled_once,
            scrambled_twice,
            "Scrambler should produce the same output for the same input.")

    def test_descrambler_reverses_scrambling(self):
        """Test if descrambler correctly reverses the scrambling process."""
        data_bits = [1, 0, 1, 1, 0, 0, 1, 0]
        seed = 93
        scrambled = scrambler(seed, data_bits)
        descrambled = descrambler(seed, scrambled)
        self.assertEqual(
            descrambled,
            data_bits,
            "Descrambler should return the original data after scrambling.")

    def test_scrambler_with_different_seeds(self):
        """Test that scrambler with different seeds produces different results."""
        data_bits = [1, 0, 1, 1, 0, 0, 1, 0]
        seed1 = 93
        seed2 = 65
        scrambled_seed1 = scrambler(seed1, data_bits)
        scrambled_seed2 = scrambler(seed2, data_bits)
        self.assertNotEqual(
            scrambled_seed1,
            scrambled_seed2,
            "Scrambler with different seeds should produce different results.")

    def test_empty_data_bits(self):
        """Test that the scrambler and descrambler can handle an empty list of data bits."""
        data_bits = []
        seed = 93
        scrambled = scrambler(seed, data_bits)
        descrambled = descrambler(seed, scrambled)
        self.assertEqual(
            scrambled,
            [],
            "Scrambling an empty list should return an empty list.")
        self.assertEqual(
            descrambled,
            [],
            "Descrambling an empty list should return an empty list.")


if __name__ == "__main__":
    unittest.main()
