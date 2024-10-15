import unittest
import numpy as np
from phy.modulation import bpsk, qpsk, qam16, qam64, OFDMmodulation, OFDMdemodulation
from phy.scrambler import scrambler, descrambler
from phy.demapper import hard_decision
from phy.bcc import bcc_encoder, bcc_decoder
from phy.interleaver import interleaver, deinterleaver
from utils.constellation_maps import (
    BPSK_CONSTELLATION,
    QPSK_CONSTELLATION,
    QAM16_CONSTELLATION,
    QAM64_CONSTELLATION
)
from utils.ofdm_helper import pad_data
# Константы для различных схем MCS
MCS_CONFIG = [
    {"modulation": "bpsk", "R": "1/2", "bpsc": 1},
    {"modulation": "qpsk", "R": "1/2", "bpsc": 2},
    # {"modulation": "qpsk", "R": "3/4", "bpsc": 2},
    # {"modulation": "qam16", "R": "1/2", "bpsc": 4},
    # {"modulation": "qam16", "R": "3/4", "bpsc": 4},
    # {"modulation": "qam64", "R": "2/3", "bpsc": 6},
    # {"modulation": "qam64", "R": "3/4", "bpsc": 6},
    # {"modulation": "qam64", "R": "5/6", "bpsc": 6},
]


class TestTxRx(unittest.TestCase):
    def setUp(self):

        self.data = np.random.randint(0, 2, 72*6*20).tolist()

    def test_tx_rx_chain(self):
        for config in MCS_CONFIG:
            with self.subTest(config=config):

                # print(config)

                scrambled_data = scrambler(127, self.data)
                # print("Scrambled data Length:", len(scrambled_data))

                # encoded_data = bcc_encoder(scrambled_data, code_rate=config["R"])

                if config["R"] == "1/2":
                    encoded_data = bcc_encoder(
                        scrambled_data, code_rate=config["R"])
                else:
                    from viterbi import Viterbi
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
                # print("Interleaved data Length:", len(interleaved_data))

                if config["modulation"] == "bpsk":
                    modulated_data = bpsk(
                        interleaved_data, constellation=False)
                elif config["modulation"] == "qpsk":
                    modulated_data = qpsk(
                        interleaved_data, constellation=False)
                elif config["modulation"] == "qam16":
                    modulated_data = qam16(
                        interleaved_data, constellation=False)
                elif config["modulation"] == "qam64":
                    modulated_data = qam64(
                        interleaved_data, constellation=False)

                # print("Mapped data Length:", len(modulated_data))

                ofdm_data = OFDMmodulation(pad_data(modulated_data))
                # print("OFDM data Length:", len(ofdm_data))

                received_demodulated_data = OFDMdemodulation(ofdm_data)
                # print("Received demodulated data Length:", len(received_demodulated_data))

                received_demodulated_data = received_demodulated_data[:len(
                    modulated_data)]
                # print("Received demodulated data Length:", len(received_demodulated_data))

                if config["modulation"] == "bpsk":
                    demodulated_data = hard_decision(
                        received_demodulated_data, BPSK_CONSTELLATION, 1)
                elif config["modulation"] == "qpsk":
                    demodulated_data = hard_decision(
                        received_demodulated_data, QPSK_CONSTELLATION, 2)
                elif config["modulation"] == "qam16":
                    demodulated_data = hard_decision(
                        received_demodulated_data, QAM16_CONSTELLATION, 4)
                elif config["modulation"] == "qam64":
                    demodulated_data = hard_decision(
                        received_demodulated_data, QAM64_CONSTELLATION, 6)

                # print("Demapped data:", demodulated_data, len(demodulated_data))
                # if interleaved_data == demodulated_data:
                #     print("Success in demapping")

                deinterleaved_data = deinterleaver(
                    demodulated_data, config["bpsc"])
                # print("Deinterleaved data:", deinterleaved_data, len(deinterleaved_data))
                # if deinterleaved_data == encoded_data:
                #     print("success in deinterleaving")

                # decoded_data = bcc_decoder(deinterleaved_data, len(self.data), code_rate=config["R"])

                if config["R"] == "1/2":
                    decoded_data = bcc_decoder(deinterleaved_data, len(
                        self.data), code_rate=config["R"])
                else:
                    from viterbi import Viterbi
                    puncture_patterns = {
                        '1/2': [1],
                        '3/4': [1, 1, 0, 1, 1, 0],
                        '2/3': [1, 1, 0, 1],
                        '4/5': [1, 1, 1, 1, 1, 0, 0, 0],
                        '5/6': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                    }
                    pattern = puncture_patterns[config["R"]]
                    dot11a_codec = Viterbi(7, [0o133, 0o171], pattern)
                    dot11a_codec.decode(deinterleaved_data)

                # print("Decoded data:", decoded_data, len(decoded_data))
                if decoded_data == scrambled_data:
                    print("success in decoding")

                descrambled_data = descrambler(127, decoded_data)
                # print("Descrambled data:", descrambled_data, len(descrambled_data))
                self.assertEqual(descrambled_data[:len(self.data)], self.data[:len(descrambled_data)],
                                 f"TxRx chain failed for MCS config: {config}")


if __name__ == "__main__":
    unittest.main()
