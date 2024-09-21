from typing import List, Callable
import numpy as np
import plotly.express as px
import pandas as pd


def bpsk(bits: List[int], constellation: bool = True) -> List[complex]:
    """
    Perform BPSK (Binary Phase Shift Keying) modulation on input bits.

    Args:
    - bits (List[int]): A list of binary values (0 or 1).
    - constellation (bool) optional: Bool parametr for constellation plot.
    Returns:
    - List[complex]: A list of complex BPSK-modulated symbols according to the map
    """
    bpsk_map = [-1, 1]
    symbols: List[complex] = []
    for bit in bits:
        symbols.append(bpsk_map[bit] + 0j)
    if constellation:
        plot_constellation(symbols, "BPSK Constellation")
    return symbols


def qpsk(bits: List[int], constellation: bool = True) -> List[complex]:
    """
    Perform QPSK (Quadrature Phase Shift Keying) modulation on input bits.
    Each pair of bits is mapped to a complex symbol.
    Args:
    - bits (List[int]): A list of binary values (0 or 1). The length of bits must be even, as two bits are needed for each QPSK symbol.
    - constellation (bool) optional: Bool parametr for constellation plot.
    Returns:
    - List[complex]: A list of complex QPSK-modulated symbols, scaled by 1/sqrt(2) according to the map:
    """
    qpsk_map = {(0, 0): (-1, -1), (0, 1): (-1, 1),
                (1, 0): (1, -1), (1, 1): (1, 1)}
    assert len(bits) % 2 == 0, "Number of bits must be even for QPSK."
    symbols: List[complex] = []
    for i in range(0, len(bits), 2):
        bit_pair = (bits[i], bits[i + 1])
        i_val, q_val = qpsk_map[bit_pair]
        symbols.append((1 / np.sqrt(2)) * (i_val + q_val * 1j))
    if constellation:
        plot_constellation(symbols, "QPSK Constellation")
    return symbols


def qam16(bits: List[int], constellation: bool = True) -> List[complex]:
    """
    Perform 16-QAM (Quadrature Amplitude Modulation) on input bits.
    Each group of 4 bits is mapped to a complex symbol.
    Args:
    - bits (List[int]): A list of binary values (0 or 1). The length of bits must be divisible by 4, as 4 bits are needed for each 16-QAM symbol.
    - constellation (bool) optional: Bool parametr for constellation plot.
    Returns:
    - List[complex]: A list of complex 16-QAM-modulated symbols, scaled by 1/sqrt(10):
      The I (real) and Q (imaginary) values are chosen from {-3, -1, 1, 3} according to the map.
    """
    qam16_map = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
    assert len(bits) % 4 == 0, "Number of bits must be divisible by 4 for 16-QAM."
    symbols: List[complex] = []
    for i in range(0, len(bits), 4):
        bit_pair_re = (bits[i], bits[i + 1])
        bit_pair_im = (bits[i + 2], bits[i + 3])
        i_val, q_val = qam16_map[bit_pair_re], qam16_map[bit_pair_im]
        symbols.append((1 / np.sqrt(10)) * (i_val + q_val * 1j))
    if constellation:
        plot_constellation(symbols, "16-QAM Constellation")
    return symbols


def qam64(bits: List[int], constellation: bool = True) -> List[complex]:
    """
    Perform 64-QAM (Quadrature Amplitude Modulation) on input bits.
    Each group of 6 bits is mapped to a complex symbol.
    Args:
    - bits (List[int]): A list of binary values (0 or 1). The length of bits must be divisible by 6, as 6 bits are needed for each 64-QAM symbol.
    - constellation (bool) optional: Bool parametr for constellation plot.
    Returns:
    - List[complex]: A list of complex 64-QAM-modulated symbols, scaled by 1/sqrt(42):
      The I (real) and Q (imaginary) values are chosen from {-7, -5, -3, -1, 1, 3, 5, 7} according to the map.
    """
    qam64_map = {
        (0, 0, 0): -7,
        (0, 0, 1): -5,
        (0, 1, 1): -3,
        (0, 1, 0): -1,
        (1, 1, 0): 1,
        (1, 1, 1): 3,
        (1, 0, 1): 5,
        (1, 0, 0): 7,
    }
    assert len(bits) % 6 == 0, "Number of bits must be divisible by 6 for 64-QAM."
    symbols: List[complex] = []
    for i in range(0, len(bits), 6):
        bit_triplet_re = (bits[i], bits[i + 1], bits[i + 2])
        bit_triplet_im = (bits[i + 3], bits[i + 4], bits[i + 5])
        i_val, q_val = qam64_map[bit_triplet_re], qam64_map[bit_triplet_im]
        symbols.append((1 / np.sqrt(42)) * (i_val + q_val * 1j))
    if constellation:
        plot_constellation(symbols, "64-QAM Constellation")
    return symbols


def plot_constellation(symbols: List[complex], title: str) -> None:
    """
    Plot the constellation diagram for the modulated symbols.

    Args:
    - symbols (List[comple]): A list of complex numbers representing modulated symbols.
    - title (str): Title of the plot.
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


def ofdm_symbol(
    d_k_n: List[int],
    p_k: List[int],
    Channel_BW: int,
    t: float,
    subcarrier_mapping: Callable,
    calculate_subcarrier_frequency_spacing: Callable,
    w_tsym: Callable,
) -> complex:
    """
    Compute the OFDM symbol based on the 17-22 formula from 802.11.2020 specification.

    Parameters:
    - d_k_n (List[int]): Data symbols on each subcarrier (array of length N_SD).
    - p_k (List[int]): Pilot symbols on each subcarrier (array of length N_ST).
    - Channel_BW (int): Channel bandwidth in MHz.
    - t (float): Time at which the symbol is evaluated.
    - w_tsym (Callable): Window function applied to the OFDM symbol.

    Returns:
    - r_data_n_t: The OFDM symbol at time t.
    """
    assert (
        len(d_k_n) == 48
    ), "The number of data subcarriers should be 48, according to Table 17-5 from 802.11.2020 specification."
    assert (
        len(p_k) == 52
    ), "The total number of subcarriers (pilot+data) should be 52, according to Table 17-5 from 802.11.2020 specification."
    assert (
        Channel_BW == 20 or Channel_BW == 10 or Channel_BW == 5
    ), "Invalid channel spacing value"

    """
    - N_SD (int): Number of data subcarriers.
    - N_ST (int): Total number of subcarriers (data + pilots).
    """

    N_SD = 48
    N_ST = 52

    delta_f = calculate_subcarrier_frequency_spacing(Channel_BW)
    # - T_FFT (float): Fast Fourier Transform period. Table 17.5 from 802.11.2020 specification.
    T_FFT = 1 / delta_f
    # - T_GI (float): Guard interval duration. Table 17.5 from 802.11.2020 specification.
    T_GI = 1 / (4 * T_FFT)
    data_sum = 0
    for k in range(N_SD):
        M_k = subcarrier_mapping(k)
        data_sum += d_k_n[k] * \
            np.exp(1j * 2 * np.pi * M_k * delta_f * (t - T_GI))

    pilot_sum = 0
    for k in range(-N_ST // 2, N_ST // 2):
        pilot_sum += p_k[k + N_ST // 2] * np.exp(
            1j * 2 * np.pi * k * delta_f * (t - T_GI)
        )

    r_data_n_t = w_tsym(t) * (data_sum + pilot_sum)

    return r_data_n_t


def subcarrier_mapping(k: int) -> int:
    """
    Function subcarrier_mapping defines mapping from the logical subcarrier number 0 to 47 into frequency offset index -26 to 26,
    while skipping the pilot subcarrier locations at the 0th (dc) subcarrier.

    Parameters:
    - k (int)- logical subcarrier number [0;47].

    Returns:
    - M (int)- frequency offset index [-26;26].
    """
    assert (
        0 <= k <= 47
    ), "The stream of complex numbers is divided into groups of N_SD = 48 complex numbers."
    if 0 <= k <= 4:
        M = k - 26
    elif 5 <= k <= 17:
        M = k - 25
    elif 18 <= k <= 23:
        M = k - 24
    elif 24 <= k <= 29:
        M = k - 29
    elif 30 <= k <= 42:
        M = k - 22
    else:
        M = k - 21
    return M


def calculate_subcarrier_frequency_spacing(Channel_BW: int) -> float:
    """
    Calculate subcarrier frequency spacing according to Table 17-5 from 802.11.2020 specification.

    Parameters:
    - Channel_BW (int): Channel bandwidth in MHz

    Returns:
    - delta_f (float): Subcarrier frequency spacing
    """
    if Channel_BW == 20:
        delta_f = 0.3125e6
    elif Channel_BW == 10:
        delta_f = 0.156e6
    elif Channel_BW == 5:
        delta_f = 0.078e6
    return delta_f


def w_tsym(t: float, T_GI: float, T_FFT: float) -> float:
    """
    Function w_tsym calculates windowing function according to 17-4 from 802.11.2020 specification.

    Parameters:
    - t (float): Time at which the symbol is evaluated.
    - T_GI (float): Guard period value.
    - T_FFT (float): Fast Fourier Transform period.

    Returns:
    - w_tsym (float): rectangular multiplication pulse
    """
    T_tr = 100e-9
    w_tsym = 1
    T = T_GI + 2 * T_FFT
    if (-1 * T_tr / 2) < t < (T_tr / 2):
        w_tsym = (np.sin((np.pi / 2) * (0.5 + t / T_tr))) ** 2
    elif (-1 * T_tr / 2) <= t < ((T - T_tr) / 2):
        w_tsym = 1
    else:
        w_tsym = (np.sin((np.pi / 2) * (0.5 - (t - T) / T_tr))) ** 2
    return float(w_tsym)
