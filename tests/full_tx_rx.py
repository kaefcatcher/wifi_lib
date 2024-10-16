import unittest
import numpy as np
from typing import List, Union

from phy.modulation import bpsk, qpsk, qam16, qam64, OFDMmodulation, OFDMdemodulation
from phy.scrambler import scrambler, descrambler
from phy.demapper import hard_decision, calculate_llr
from phy.bcc import bcc_encoder, bcc_decoder
from phy.interleaver import interleaver, deinterleaver
from phy.LDPC import (
    rref, create_generator_matrix, encode_with_ldpc, recover_information_bits
)
from phy.SPA import SPA
from utils.constellation_maps import (
    BPSK_CONSTELLATION,
    QPSK_CONSTELLATION,
    QAM16_CONSTELLATION,
    QAM64_CONSTELLATION
)
from utils.ofdm_helper import pad_data
from utils.LDPC_matrix import LDPCMatrixGenerator
from viterbi import Viterbi


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


class TestTxRx(unittest.TestCase):
    def setUp(self):
        self.data = np.random.randint(0, 2, 72 * 6 * 20).tolist()

    def test_tx_rx_chain(self):
        for config in MCS_CONFIG:
            with self.subTest(config=config):
                scrambled_data = scrambler(127, self.data)

                if config["R"] == "1/2":
                    encoded_data = bcc_encoder(
                        scrambled_data, code_rate=config["R"])
                else:
                    puncture_patterns = {
                        '1/2': [1],
                        '3/4': [1, 1, 0, 1, 1, 0],
                        '2/3': [1, 1, 0, 1],
                        '5/6': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                    }
                    pattern = puncture_patterns[config["R"]]
                    dot11a_codec = Viterbi(7, [0o133, 0o171], pattern)
                    encoded_data = dot11a_codec.encode(scrambled_data)

                print("Encoded data Length:", len(encoded_data))

                interleaved_data = interleaver(encoded_data, config["bpsc"])

                modulation_functions = {
                    "bpsk": bpsk,
                    "qpsk": qpsk,
                    "qam16": qam16,
                    "qam64": qam64
                }

                modulated_data = modulation_functions[config["modulation"]](interleaved_data, constellation=False)

                ofdm_data = OFDMmodulation(pad_data(modulated_data))

                received_demodulated_data = OFDMdemodulation(ofdm_data)

                received_demodulated_data = received_demodulated_data[:len(modulated_data)]

                constellation_map = {
                    "bpsk": (BPSK_CONSTELLATION, 1),
                    "qpsk": (QPSK_CONSTELLATION, 2),
                    "qam16": (QAM16_CONSTELLATION, 4),
                    "qam64": (QAM64_CONSTELLATION, 6)
                }

                constellation, bits_per_symbol = constellation_map[config["modulation"]]
                demodulated_data = hard_decision(received_demodulated_data, constellation, bits_per_symbol)

                deinterleaved_data = deinterleaver(demodulated_data, config["bpsc"])

                if config["R"] == "1/2":
                    decoded_data = bcc_decoder(deinterleaved_data, len(self.data), code_rate=config["R"])
                else:
                    puncture_patterns = {
                        '1/2': [1],
                        '3/4': [1, 1, 0, 1, 1, 0],
                        '2/3': [1, 1, 0, 1],
                        '4/5': [1, 1, 1, 1, 1, 0, 0, 0],
                        '5/6': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                    }
                    pattern = puncture_patterns[config["R"]]
                    dot11a_codec = Viterbi(7, [0o133, 0o171], pattern)
                    decoded_data = dot11a_codec.decode(deinterleaved_data)

                if decoded_data == scrambled_data:
                    print("Success in decoding")

                descrambled_data = descrambler(127, decoded_data)

                self.assertEqual(descrambled_data[:len(self.data)], self.data[:len(descrambled_data)],
                                 f"TxRx chain failed for MCS config: {config}")


if __name__ == "__main__":
    unittest.main()
