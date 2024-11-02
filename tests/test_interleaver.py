import unittest
from typing import List
from phy.Interleaving import Interleaving
import numpy as np

class TestInterleaving(unittest.TestCase):
    def setUp(self):
        # This will run before each test method
        self.sample_data = np.random.randint(0, 2, 72 * 6 * 20).tolist() # Example input data
        self.N_bpcs = 1  # Number of coded bits per subcarrier for testing

    def test_interleaver_and_deinterleaver(self):
        # Perform interleaving
        interleaved_data = Interleaving.interleaver(self.sample_data, self.N_bpcs)
        # Perform deinterleaving
        deinterleaved_data = Interleaving.deinterleaver(interleaved_data, self.N_bpcs)

        # Check if deinterleaving the interleaved data returns the original data
        self.assertEqual(deinterleaved_data, self.sample_data, "Deinterleaved data does not match original data")

    def test_interleaver_output_length(self):
        # Check if the length of interleaved data matches the original data length
        interleaved_data = Interleaving.interleaver(self.sample_data, self.N_bpcs)
        self.assertEqual(len(interleaved_data), len(self.sample_data), "Interleaved data length mismatch")

    def test_deinterleaver_output_length(self):
        # Check if the length of deinterleaved data matches the original data length
        interleaved_data = Interleaving.interleaver(self.sample_data, self.N_bpcs)
        deinterleaved_data = Interleaving.deinterleaver(interleaved_data, self.N_bpcs)
        self.assertEqual(len(deinterleaved_data), len(self.sample_data), "Deinterleaved data length mismatch")


if __name__ == "__main__":
    unittest.main()
