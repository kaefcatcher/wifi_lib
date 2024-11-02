from typing import List
import numpy as np

class OFDM:
    def __init__(self, num_subcarriers: int = 64):
        self.num_subcarriers = num_subcarriers
        self.pilot_positions = [-21, -7, 7, 21]
        self.zero_position = 0
        self.data_positions = [i for i in range(-num_subcarriers//2 + 1, num_subcarriers//2)
                               if i not in self.pilot_positions and i != self.zero_position]

    def modulation(self, data: List[complex]) -> List[List[complex]]:
        ofdm_frames = []

        padded_data = np.pad(data, (0, len(self.data_positions) - len(data) % len(self.data_positions)), 'constant')

        for start_idx in range(0, len(padded_data), len(self.data_positions)):
            ofdm_frame:List[complex] = [0] * self.num_subcarriers
            sub_frame_data = padded_data[start_idx:start_idx + len(self.data_positions)]

            for idx, pos in enumerate(self.data_positions):
                ofdm_frame[pos + self.num_subcarriers//2] = sub_frame_data[idx]

            pilot_value = 1.0 + 0j
            for pos in self.pilot_positions:
                ofdm_frame[pos + self.num_subcarriers//2] = pilot_value

            ofdm_frames.append(ofdm_frame)

        return ofdm_frames

    def demodulation(self, ofdm_frames: List[List[complex]]) -> List[complex]:
        """
        Demodulate OFDM frames back into a stream of complex symbols.

        Parameters:
        - ofdm_frames (List[List[complex]]): A list of OFDM frames, where each frame is a list of complex numbers.

        Returns:
        - List[complex]: A list of demodulated complex symbols.
        """
        data = []

        for ofdm_frame in ofdm_frames:
            frame_data = [ofdm_frame[pos + self.num_subcarriers//2] for pos in self.data_positions]
            data.extend(frame_data)

        return data
