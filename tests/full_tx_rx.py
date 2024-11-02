import unittest
import numpy as np
from typing import List, Union

from phy.Mapping import Mapping
from phy.OFDM import OFDM
from phy.Scrambling import Scrambling
# from phy.demapper import hard_decision, calculate_llr
from phy.BCC import ViterbiCodec
from phy.Interleaving import Interleaving
# from phy.LDPC import LDPC
# from phy.SPA import SPA
from utils.constellation_maps import (
    BPSK_CONSTELLATION,
    QPSK_CONSTELLATION,
    QAM16_CONSTELLATION,
    QAM64_CONSTELLATION
)
from utils.LDPC_matrix import LDPCMatrixGenerator
from utils.bit2str import list2string
# from viterbi import Viterbi


MCS_CONFIG = [
    {"modulation": "bpsk", "R": "1/2", "bpsc": 1},
    {"modulation": "qpsk", "R": "1/2", "bpsc": 2},
    {"modulation": "qpsk", "R": "3/4", "bpsc": 2},
    {"modulation": "qam16", "R": "1/2", "bpsc": 4},
    {"modulation": "qam16", "R": "3/4", "bpsc": 4},
    {"modulation": "qam64", "R": "2/3", "bpsc": 6},
    {"modulation": "qam64", "R": "3/4", "bpsc": 6},
    {"modulation": "qam64", "R": "5/6", "bpsc": 6},
]

puncture_patterns = {
        '1/2': [1],
        '3/4': [1, 1, 0, 1, 1, 0],
        '2/3': [1, 1, 0, 1],
        '5/6': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}


class TestTxRx(unittest.TestCase):
    def setUp(self):
        self.data = np.random.randint(0, 2, 72 * 6 * 20).tolist()

    def test_tx_rx_chain(self):
        for config in MCS_CONFIG:
            with self.subTest(config=config):
                data = self.data
                pattern = puncture_patterns[config["R"]]
                scrambler = Scrambling(127)
                coder = ViterbiCodec(7,[0o133,0o171],list2string(pattern))
                ofdm = OFDM(64)
                mapper = Mapping()
                modulation_functions = {
                                    "bpsk": mapper.bpsk,
                                    "qpsk": mapper.qpsk,
                                    "qam16": mapper.qam16,
                                    "qam64": mapper.qam64
                }
                interleaver = Interleaving()

                scrambled_data = scrambler.scramble(data)


                encoded_data = [int(bit) for bit in (coder.encode(list2string(scrambled_data)))]

                interleaved_data = interleaver.interleaver(encoded_data, config["bpsc"])

                modulated_data = modulation_functions[config["modulation"]](interleaved_data, constellation=False)

                ofdm_data = ofdm.modulation(modulated_data)

                received_demodulated_data = ofdm.demodulation(ofdm_data)
                received_demodulated_data = received_demodulated_data[:len(modulated_data)]

                constellation_map = {
                    "bpsk": (BPSK_CONSTELLATION, 1),
                    "qpsk": (QPSK_CONSTELLATION, 2),
                    "qam16": (QAM16_CONSTELLATION, 4),
                    "qam64": (QAM64_CONSTELLATION, 6)
                }

                constellation, bits_per_symbol = constellation_map[config["modulation"]]
                demodulated_data = mapper.hard_decision(received_demodulated_data, constellation, bits_per_symbol)

                deinterleaved_data = interleaver.deinterleaver(demodulated_data, config["bpsc"])

                decoded_data_str = coder.decode(list2string(deinterleaved_data))
                decoded_data = [int(bit) for bit in decoded_data_str]

                # # if decoded_data == scrambled_data:
                # #     print("Success in decoding")

                descrambled_data = scrambler.descramble(decoded_data)


                self.assertEqual(descrambled_data[:len(data)], data[:len(descrambled_data)],
                                 f"TxRx chain failed for MCS config: {config}")


if __name__ == "__main__":
    unittest.main()
