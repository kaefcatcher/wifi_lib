# import unittest
# import numpy as np

# from wifi_lib.phy.modulation import (
#     bpsk, qpsk, qam16, qam64, OFDMdemodulation, OFDMmodulation
# )
# from wifi_lib.phy.scrambler import (
#     scrambler, descrambler
# )
# from wifi_lib.phy.demapper import (
#     approximate_soft_decision_llr, approximate_llr, soft_decision_llr, calculate_llr, hard_decision
# )

# from wifi_lib.phy.bcc import (
#     bcc_encoder, bcc_decoder
# )

# from wifi_lib.phy.interleaver import (
#     interleaver, deinterleaver
# )


# class TestModulation(unittest.TestCase):
#     def test_bpsk(self):
#         # Test BPSK modulation with a simple bit sequence
#         bits = [0, 1, 0, 1]
#         expected_symbols = [-1 + 0j, 1 + 0j, -1 + 0j, 1 + 0j]
#         result = bpsk(bits, constellation=False)
#         self.assertEqual(result, expected_symbols)

#     def test_qpsk(self):
#         # Test QPSK modulation with a simple bit sequence
#         bits = [0, 0, 0, 1, 1, 0, 1, 1]
#         expected_symbols = [
#             (-1 - 1j) / (2**0.5),
#             (-1 + 1j) / (2**0.5),
#             (1 - 1j) / (2**0.5),
#             (1 + 1j) / (2**0.5),
#         ]
#         result = qpsk(bits, constellation=False)
#         self.assertEqual(result, expected_symbols)

#     def test_qpsk_invalid_bits(self):
#         # Test that an error is raised for an odd number of bits in QPSK
#         bits = [0, 1, 0]
#         with self.assertRaises(AssertionError):
#             qpsk(bits, constellation=False)

#     def test_qam16(self):
#         # Test 16-QAM modulation with a simple bit sequence
#         bits = [0, 0, 0, 0, 1, 1, 1, 1]
#         expected_symbols = [(-3 - 3j) / (10**0.5), (1 + 1j) / (10**0.5)]
#         result = qam16(bits, constellation=False)
#         self.assertEqual(result, expected_symbols)

#     def test_qam16_invalid_bits(self):
#         # Test that 16-QAM raises assertion error for incorrect number of bits
#         bits = [0, 1, 0]
#         with self.assertRaises(AssertionError):
#             qam16(bits, constellation=False)

#     def test_qam64(self):
#         # Test 64-QAM modulation with a simple bit sequence
#         bits = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
#         expected_symbols = [(-7 - 7j) / (42**0.5), (3 + 3j) / (42**0.5)]
#         result = qam64(bits, constellation=False)

#         for res, exp in zip(result, expected_symbols):
#             self.assertTrue(
#                 np.isclose(res, exp, rtol=1e-5, atol=1e-8),
#                 msg=f"Result {res} is not close to expected {exp}",
#             )

#     def test_qam64_invalid_bits(self):
#         # Test that 64-QAM raises assertion error for incorrect number of bits
#         bits = [0, 1, 0]
#         with self.assertRaises(AssertionError):
#             qam64(bits, constellation=False)

#     def test_scrambler_deterministic(self):
#         """Test if scrambler is deterministic (same input produces same output)."""
#         data_bits = [1, 0, 1, 1, 0, 0, 1, 0]
#         seed = 93
#         scrambled_once = scrambler(seed, data_bits)
#         scrambled_twice = scrambler(seed, data_bits)
#         self.assertEqual(
#             scrambled_once,
#             scrambled_twice,
#             "Scrambler should produce the same output for the same input.")

#     def test_descrambler_reverses_scrambling(self):
#         """Test if descrambler correctly reverses the scrambling process."""
#         data_bits = [1, 0, 1, 1, 0, 0, 1, 0]
#         seed = 93
#         scrambled = scrambler(seed, data_bits)
#         descrambled = descrambler(seed, scrambled)
#         self.assertEqual(
#             descrambled,
#             data_bits,
#             "Descrambler should return the original data after scrambling.")

#     def test_scrambler_with_different_seeds(self):
#         """Test that scrambler with different seeds produces different results."""
#         data_bits = [1, 0, 1, 1, 0, 0, 1, 0]
#         seed1 = 93
#         seed2 = 65
#         scrambled_seed1 = scrambler(seed1, data_bits)
#         scrambled_seed2 = scrambler(seed2, data_bits)
#         self.assertNotEqual(
#             scrambled_seed1,
#             scrambled_seed2,
#             "Scrambler with different seeds should produce different results.")

#     def test_empty_data_bits(self):
#         """Test that the scrambler and descrambler can handle an empty list of data bits."""
#         data_bits = []
#         seed = 93
#         scrambled = scrambler(seed, data_bits)
#         descrambled = descrambler(seed, scrambled)
#         self.assertEqual(
#             scrambled,
#             [],
#             "Scrambling an empty list should return an empty list.")
#         self.assertEqual(
#             descrambled,
#             [],
#             "Descrambling an empty list should return an empty list.")

#     def setUp(self):
#         self.constellation_0 = [1 + 1j, 1 - 1j]
#         self.constellation_1 = [-1 + 1j, -1 - 1j]
#         self.symbols = [1 + 0.5j, -1 - 0.5j, 0 + 0j]
#         self.sigma = 1.0

#     def test_calculate_llr(self):
#         symbol = 1 + 0.5j

#         p_b0 = sum(np.exp(-np.abs(symbol - d0) ** 2 / (2 * self.sigma ** 2))
#                    for d0 in self.constellation_0)
#         p_b1 = sum(np.exp(-np.abs(symbol - d1) ** 2 / (2 * self.sigma ** 2))
#                    for d1 in self.constellation_1)

#         expected_llr = np.log(p_b0 / p_b1)

#         result = calculate_llr(
#             symbol,
#             self.constellation_0,
#             self.constellation_1,
#             self.sigma)
#         self.assertAlmostEqual(result, expected_llr, places=5)

#     def test_soft_decision_llr(self):
#         expected_llrs = [calculate_llr(symbol, self.constellation_0, self.constellation_1, self.sigma)
#                          for symbol in self.symbols]
#         result = soft_decision_llr(
#             self.symbols,
#             self.constellation_0,
#             self.constellation_1,
#             self.sigma)
#         self.assertEqual(len(result), len(self.symbols))
#         for r, expected in zip(result, expected_llrs):
#             self.assertAlmostEqual(r, expected, places=5)

#     def test_approximate_llr(self):
#         symbol = 1 + 0.5j
#         expected_llr = (-1 / (2 * self.sigma**2)) * (min([(abs(symbol - p))**2 for p in self.constellation_0]) -
#                                                      min([(abs(symbol - p))**2 for p in self.constellation_1]))
#         result = approximate_llr(
#             symbol,
#             self.constellation_0,
#             self.constellation_1,
#             self.sigma)
#         self.assertAlmostEqual(result, expected_llr, places=5)

#     def test_approximate_soft_decision_llr(self):
#         expected_llrs = [approximate_llr(symbol, self.constellation_0, self.constellation_1, self.sigma)
#                          for symbol in self.symbols]
#         result = approximate_soft_decision_llr(
#             self.symbols,
#             self.constellation_0,
#             self.constellation_1,
#             self.sigma)
#         self.assertEqual(len(result), len(self.symbols))
#         for r, expected in zip(result, expected_llrs):
#             self.assertAlmostEqual(r, expected, places=5)

#     def test_interleaver_and_deinterleaver(self):
#         data = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
#         N_bpscs = 2  # Example: QPSK (2 bits per subcarrier)
#         i_ss = 1     # Single stream

#         interleaved_data = interleaver(N_bpscs, i_ss, data)
#         deinterleaved_data = deinterleaver(N_bpscs, i_ss, interleaved_data)

#         self.assertEqual(deinterleaved_data, data,
#                          "Deinterleaver output does not match original data")

#     def test_bcc_encoder_decoder(self):
#         # Define test data
#         data_bits = [1, 0, 1, 1, 0, 0, 1, 0]

#         # Encode data
#         encoded_bits = bcc_encoder(data_bits)

#         # Decode encoded data
#         decoded_bits = bcc_decoder(
#             encoded_bits, original_data_len=len(data_bits))

#         # Check if decoded data matches original data
#         self.assertEqual(decoded_bits, data_bits,
#                          "Decoded bits do not match original data bits")

#     def test_bcc_encoder_decoder_with_all_zeros(self):
#         # Define all zeros data
#         data_bits = [0] * 8

#         # Encode data
#         encoded_bits = bcc_encoder(data_bits)

#         # Decode encoded data
#         decoded_bits = bcc_decoder(
#             encoded_bits, original_data_len=len(data_bits))

#         # Check if decoded data matches original data
#         self.assertEqual(decoded_bits, data_bits,
#                          "Decoded bits do not match all-zero original data")

#     def test_bcc_encoder_decoder_with_all_ones(self):
#         # Define all ones data
#         data_bits = [1] * 8

#         # Encode data
#         encoded_bits = bcc_encoder(data_bits)

#         # Decode encoded data
#         decoded_bits = bcc_decoder(
#             encoded_bits, original_data_len=len(data_bits))

#         # Check if decoded data matches original data
#         self.assertEqual(decoded_bits, data_bits,
#                          "Decoded bits do not match all-one original data")


# if __name__ == "__main__":
#     unittest.main()
