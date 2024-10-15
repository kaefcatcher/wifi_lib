from typing import List
import numpy as np


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


def calculate_llr(
        symbol: complex, constellation_0: List[complex], constellation_1: List[complex], sigma: float) -> float:
    """
    Calculate the Log-Likelihood Ratio (LLR) for a single bit based on the given received symbol.

    Parameters:
    - symbol (complex): The received complex symbol.
    - constellation_0 (List[complex]): The constellation points where the bit is 0.
    - constellation_1 (List[complex]): The constellation points where the bit is 1.
    - sigma (float): The standard deviation of the noise (sigma^2 is the noise variance).

    Returns:
    - float: The calculated LLR value for the bit.
    """
    def gaussian_probability(s, d, sigma):
        return np.exp(-np.abs(s - d)**2 / (2 * sigma**2))

    p_b0 = sum(gaussian_probability(symbol, d0, sigma)
               for d0 in constellation_0)

    p_b1 = sum(gaussian_probability(symbol, d1, sigma)
               for d1 in constellation_1)

    llr = np.log(p_b0 / p_b1)
    return llr


def soft_decision_llr(symbols: List[complex], constellation_0: List[complex],
                      constellation_1: List[complex], sigma: float) -> List[float]:
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
    return [calculate_llr(symbol, constellation_0,
                          constellation_1, sigma) for symbol in symbols]


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
    return [approximate_llr(symbol, constellation_0,
                            constellation_1, sigma) for symbol in symbols]
