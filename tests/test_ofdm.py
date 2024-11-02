import unittest
from typing import List
import numpy as np
from phy.OFDM import OFDM # Replace 'ofdm' with the actual module name if different

class TestOFDM(unittest.TestCase):
    def setUp(self):
        self.num_subcarriers = 64
        self.ofdm = OFDM(self.num_subcarriers)

    def test_modulation_frame_structure(self):
        # Test if modulation fills OFDM frame correctly with data and pilot values
        data = [complex(i, i) for i in range(len(self.ofdm.data_positions))]
        ofdm_frames = self.ofdm.modulation(data)

        for frame in ofdm_frames:
            # Check data positions contain the data
            frame_data = [frame[pos + self.num_subcarriers // 2] for pos in self.ofdm.data_positions]
            self.assertEqual(frame_data, data[:len(self.ofdm.data_positions)])

            # Check pilot positions contain the pilot value
            for pos in self.ofdm.pilot_positions:
                self.assertEqual(frame[pos + self.num_subcarriers // 2], 1.0 + 0j, "Pilot value incorrect")

    def test_modulation_padding(self):
        # Test if modulation correctly pads data that doesn't fill an entire frame
        data = [complex(i, i) for i in range(len(self.ofdm.data_positions) - 5)]  # Not a full frame of data
        ofdm_frames = self.ofdm.modulation(data)

        # Last frame should have padding at the end
        last_frame = ofdm_frames[-1]
        frame_data = [last_frame[pos + self.num_subcarriers // 2] for pos in self.ofdm.data_positions]

        # Check that padded values are zero
        self.assertEqual(frame_data[len(data):], [0] * (len(self.ofdm.data_positions) - len(data)))

    def test_modulation_demodulation(self):
        # Test if demodulating a modulated frame returns the original data
        original_data = [complex(i, i) for i in range(100)]
        ofdm_frames = self.ofdm.modulation(original_data)
        demodulated_data = self.ofdm.demodulation(ofdm_frames)

        # Check that the demodulated data matches the original data (excluding any padding)
        self.assertEqual(demodulated_data[:len(original_data)], original_data, "Demodulation did not return original data correctly")

    def test_demodulation_empty(self):
        # Test demodulation on an empty list of frames
        demodulated_data = self.ofdm.demodulation([])
        self.assertEqual(demodulated_data, [], "Demodulating empty frames should return an empty list")

if __name__ == "__main__":
    unittest.main()
