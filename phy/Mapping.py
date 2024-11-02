from typing import List, Union
import numpy as np
import plotly.express as px
import pandas as pd


from utils.constellation_maps import (
    BPSK_CONSTELLATION,
    QPSK_CONSTELLATION,
    QAM16_CONSTELLATION,
    QAM64_CONSTELLATION
)

class Mapping:
    @staticmethod
    def bpsk(bits: Union[List[int], List[complex]], constellation: bool = True) -> List[complex]:
        """
        Perform BPSK (Binary Phase Shift Keying) modulation on input bits.

        Parameters:
        - bits (List[int]): original bits to map.
        - constellation (bool): constellation map visualisation on a plot (True by default).

        Returns:
        - symbols (List[complex]): mapped bits.
        """
        symbols: List[complex] = []
        for bit in bits:
            symbols.append(BPSK_CONSTELLATION[bit])
        if constellation:
            Mapping.plot_constellation(symbols, "BPSK Constellation")
        return symbols

    @staticmethod
    def qpsk(bits: List[int], constellation: bool = True) -> List[complex]:
        """
        Perform QPSK (Quadrature Phase Shift Keying) modulation on input bits.

        Parameters:
        - bits (List[int]): original bits to map.
        - constellation (bool): constellation map visualisation on a plot (True by default).

        Returns:
        - symbols (List[complex]): mapped bits.
        """
        assert len(bits) % 2 == 0, "Number of bits must be even for QPSK."
        symbols: List[complex] = []
        for i in range(0, len(bits), 2):
            index = bits[i] * 2 + bits[i + 1]
            symbols.append(QPSK_CONSTELLATION[index])
        if constellation:
            Mapping.plot_constellation(symbols, "QPSK Constellation")
        return symbols

    @staticmethod
    def qam16(bits: List[int], constellation: bool = True) -> List[complex]:
        """
        Perform 16-QAM (Quadrature Amplitude Modulation) on input bits.

        Parameters:
        - bits (List[int]): original bits to map.
        - constellation (bool): constellation map visualization on a plot (True by default).

        Returns:
        - symbols (List[complex]): mapped bits.
        """
        assert len(bits) % 4 == 0, "Number of bits must be divisible by 4 for 16-QAM."
        symbols: List[complex] = []

        for i in range(0, len(bits), 4):

            index = (bits[i] << 3) | (bits[i + 1] << 2) | (bits[i + 2] << 1) | bits[i + 3]
            symbols.append(QAM16_CONSTELLATION[index])

        if constellation:
            Mapping.plot_constellation(symbols, "16-QAM Constellation")

        return symbols

    @staticmethod
    def qam64(bits: List[int], constellation: bool = True) -> List[complex]:
        """
        Perform 64-QAM (Quadrature Amplitude Modulation) on input bits.

        Parameters:
        - bits (List[int]): original bits to map.
        - constellation (bool): constellation map visualization on a plot (True by default).

        Returns:
        - symbols (List[complex]): mapped bits.
        """
        assert len(bits) % 6 == 0, "Number of bits must be divisible by 6 for 64-QAM."
        symbols: List[complex] = []

        for i in range(0, len(bits), 6):

            index = (bits[i] << 5) | (bits[i + 1] << 4) | (bits[i + 2] << 3) | (bits[i + 3] << 2) | (bits[i + 4] << 1) | bits[i + 5]
            symbols.append(QAM64_CONSTELLATION[index])

        if constellation:
            Mapping.plot_constellation(symbols, "64-QAM Constellation")

        return symbols

    @staticmethod
    def plot_constellation(symbols: List[complex], title: str) -> None:
        """
        Plot the constellation diagram for the modulated symbols.

        Parameters:
        - symbols (List[comple]): A list of complex numbers representing modulated symbols.
        - title (str): Title of the plot.

        Returns:
        - None
        """
        df = pd.DataFrame(
            {
                "Real": [symbol.real for symbol in symbols],
                "Imag": [symbol.imag for symbol in symbols],
            }
        )

        fig = px.scatter(
            df,
            x="Real",
            y="Imag",
            title=title,
            labels={"Real": "In-phase (I)", "Imag": "Quadrature (Q)"},
        )
        fig.update_layout(width=700, height=700)

        fig.show()

    @staticmethod
    def hard_decision(symbols: List[complex],
                      constellation: List[complex],
                      bits_per_symbol: int) -> List[int]:
        """
        Hard decision demapping: find the closest constellation point for each symbol.

        Parameters:
        - symbols (List[complex]): The list of received complex symbols.
        - constellation (List[complex]): The list of constellation points.
        - bits_per_symbol (int): Number of bits represented by each symbol.

        Returns:
        - List[int]: The list of demapped bits based on the closest point.
        """
        demapped_indices = []
        for symbol in symbols:
            distances = [abs(symbol - point) for point in constellation]
            demapped_indices.append(np.argmin(distances))

        demapped_bits = []
        for index in demapped_indices:
            bits = [(index >> j) & 1 for j in reversed(range(bits_per_symbol))]
            demapped_bits.extend(bits)

        return demapped_bits

    @staticmethod
    def calculate_llr(s, constellation, noise_variance, modulation_order):
        """
        Calculate LLR for a given received symbol based on constellation and modulation order.

        :param s: Received symbol (complex number).
        :param constellation: List of constellation points (complex numbers).
        :param noise_variance: Noise variance (sigma^2).
        :param modulation_order: Number of bits per symbol for the modulation scheme (e.g., 1 for BPSK, 2 for QPSK, 4 for QAM-16).
        :return: LLR value for each bit.
        """
        llr_values = np.zeros(modulation_order)

        for i in range(modulation_order):
            # Symbols for bit 0 and bit 1
            d_0 = np.array([constellation[j] for j in range(2**modulation_order) if (j & (1 << (modulation_order - 1 - i))) == 0])
            d_1 = np.array([constellation[j] for j in range(2**modulation_order) if (j & (1 << (modulation_order - 1 - i))) != 0])

            numerator = np.sum((1 / np.sqrt(2 * np.pi * noise_variance)) *
                               np.exp(-np.abs(s - d_0) ** 2 / (2 * noise_variance)))

            denominator = np.sum((1 / np.sqrt(2 * np.pi * noise_variance)) *
                                 np.exp(-np.abs(s - d_1) ** 2 / (2 * noise_variance)))

            llr_values[i] = np.log(numerator / denominator) if denominator != 0 else float('inf')

        return llr_values

    @staticmethod
    def approximate_llr(
            symbol: complex, constellation_0: List[complex], constellation_1: List[complex], sigma: float) -> float:
        """
        Approximate the Log-Likelihood Ratio (LLR) for a single bit based on the given received symbol.

        Parameters:
        - symbol (complex): The received complex symbol.
        - constellation_0 (List[complex]): The constellation points where the bit is 0.
        - constellation_1 (List[complex]): The constellation points where the bit is 1.
        - sigma (float): The standard deviation of the noise (sigma^2 is the noise variance).

        Returns:
        - float: The calculated LLR value for the bit.
        """
        d0 = [(abs(symbol - point))**2 for point in constellation_0]
        d1 = [(abs(symbol - point))**2 for point in constellation_1]

        llr = (-1 / 2 * sigma**2) * (min(d0) - min(d1))
        return llr

    @staticmethod
    def approximate_soft_decision_llr(
            symbols: List[complex], constellation_0: List[complex], constellation_1: List[complex], sigma: float) -> List[float]:
        """
        Perform soft decision demapping using LLR for a list of received symbols.

        Parameters:
        - symbols (List[complex]): The list of received complex symbols.
        - constellation_0 (List[complex]): The constellation points where the bit is 0.
        - constellation_1 (List[complex]): The constellation points where the bit is 1.
        - sigma (float): The standard deviation of the noise (sigma^2 is the noise variance).

        Returns:
        - List[float]: A list of LLR values for each received symbol.
        """
        return [Mapping.approximate_llr(symbol, constellation_0,
                                constellation_1, sigma) for symbol in symbols]
