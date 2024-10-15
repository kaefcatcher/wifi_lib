from typing import List
from typing import List, Callable, Union
import numpy as np
import plotly.express as px
import pandas as pd


from utils.constellation_maps import (
    BPSK_CONSTELLATION,
    QPSK_CONSTELLATION,
    QAM16_CONSTELLATION,
    QAM64_CONSTELLATION
)


def bpsk(bits: List[int], constellation: bool = True) -> List[complex]:
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
        plot_constellation(symbols, "BPSK Constellation")
    return symbols


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
        plot_constellation(symbols, "QPSK Constellation")
    return symbols


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
        # Convert the 4 bits into a single index for the constellation
        index = (bits[i] << 3) | (bits[i + 1] <<
                                  2) | (bits[i + 2] << 1) | bits[i + 3]
        symbols.append(QAM16_CONSTELLATION[index])

    if constellation:
        plot_constellation(symbols, "16-QAM Constellation")

    return symbols


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
        # Convert the 6 bits into a single index for the constellation
        index = (bits[i] << 5) | (bits[i + 1] << 4) | (bits[i + 2] <<
                                                       3) | (bits[i + 3] << 2) | (bits[i + 4] << 1) | bits[i + 5]
        symbols.append(QAM64_CONSTELLATION[index])

    if constellation:
        plot_constellation(symbols, "64-QAM Constellation")

    return symbols


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


def OFDMmodulation(data: List[complex], num_subcarriers: int = 64) -> List[List[complex]]:
    pilot_positions = [-21, -7, 7, 21]
    zero_position = 0

    data_positions = [i for i in range(-num_subcarriers//2 + 1, num_subcarriers//2)
                      if i not in pilot_positions and i != zero_position]
    ofdm_frames = []

    padded_data = np.pad(data, (0, len(data_positions) -
                         len(data) % len(data_positions)), 'constant')

    for start_idx in range(0, len(padded_data), len(data_positions)):
        ofdm_frame = [0] * num_subcarriers
        sub_frame_data = padded_data[start_idx:start_idx + len(data_positions)]

        for idx, pos in enumerate(data_positions):
            ofdm_frame[pos + num_subcarriers//2] = sub_frame_data[idx]

        pilot_value = 1.0 + 0j
        for pos in pilot_positions:
            ofdm_frame[pos + num_subcarriers//2] = pilot_value

        ofdm_frames.append(ofdm_frame)

    return ofdm_frames


def OFDMdemodulation(ofdm_frames: List[List[complex]], num_subcarriers: int = 64) -> List[complex]:
    data = []
    pilot_positions = [-21, -7, 7, 21]
    zero_position = 0

    data_positions = [i for i in range(-num_subcarriers//2 + 1, num_subcarriers//2)
                      if i not in pilot_positions and i != zero_position]

    for ofdm_frame in ofdm_frames:
        frame_data = [ofdm_frame[pos + num_subcarriers//2]
                      for pos in data_positions]
        data.extend(frame_data)

    return data


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
