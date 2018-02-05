# Copyright (C) 2017-2018 Michael Freitag, Shahin Amiriparian, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller
#
# This file is part of auDeep.
#
# auDeep is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# auDeep is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep. If not, see <http://www.gnu.org/licenses/>.

"""Spectrogram utilities"""
import numpy as np
from scipy.signal import spectrogram, hann


def pre_emphasis_filter(signal: np.ndarray,
                        alpha: float = 0.95) -> np.ndarray:
    """
    Applies a pre-emphasis filter with the specified coefficient to the signal.
        
    Parameters
    ----------
    signal: numpy.ndarray
        The signal to which the filter should be applied.
    alpha: float
        The coefficient of the filter

    Returns
    -------
    numpy.ndarray
        The filtered signal
    """
    # noinspection PyTypeChecker
    return np.append(signal[0], signal[1:] - signal[:-1] * alpha)


def power_spectrum(signal: np.ndarray,
                   fs: int,
                   window_width: int,
                   window_overlap: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Computes the power spectrum of the specified signal.
    
    A periodic Hann window with the specified width and overlap is used.
    
    Parameters
    ----------
    signal: numpy.ndarray
        The input signal
    fs: int
        Sampling frequency of the input signal
    window_width: int
        Width of the Hann windows in samples
    window_overlap: int
        Overlap between Hann windows in samples

    Returns
    -------
    f: numpy.ndarray
        Array of frequency values for the first axis of the returned spectrogram
    t: numpy.ndarray
        Array of time values for the second axis of the returned spectrogram
    sxx: numpy.ndarray
        Power spectrogram of the input signal with axes [frequency, time]
    """
    f, t, sxx = spectrogram(x=signal,
                            fs=fs,
                            window=hann(window_width, sym=False),
                            noverlap=window_overlap,
                            mode="magnitude")

    return f, t, (1.0 / window_width) * (sxx ** 2)


def mel_filter_bank(fs: int,
                    window_width: int,
                    n_filt: int = 40) -> (np.ndarray, np.ndarray):
    """
    Computes Mel filter banks for the specified parameters.
    
    A power spectrogram can be converted to a Mel spectrogram by multiplying it with the filter bank. This method exists
    so that the computation of Mel filter banks does not have to be repeated for each computation of a Mel spectrogram.
    
    The coefficients of Mel filter banks depend on the sampling frequency and the window width that were used to 
    generate power spectrograms.
    
    Parameters
    ----------
    fs: int
        The sampling frequency of signals
    window_width: int
        The window width in samples used to generate spectrograms
    n_filt: int
        The number of filters to compute

    Returns
    -------
    f: numpy.ndarray
        Array of Hertz frequency values for the filter banks
    filters: numpy.ndarray
        Array of Mel filter bank coefficients. The first axis corresponds to different filters, and the second axis
        corresponds to the original frequency bands
    """
    n_fft = window_width
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((n_fft + 1) * hz_points / fs)

    fbank = np.zeros((n_filt, int(np.floor(n_fft / 2 + 1))))

    for m in range(1, n_filt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return hz_points[1:n_filt + 1], fbank


def mel_spectrum(power_spectrum: np.ndarray,
                 mel_fbank: np.ndarray = None,
                 fs: int = None,
                 window_width: int = None,
                 n_filt: int = 40) -> np.ndarray:
    """
    Computes a Mel spectrogram from the specified power spectrogram.
    
    Optionally, precomputed Mel filter banks can be passed to this function, in which case the n_filt, fs, and 
    window_width parameters are ignored. If precomputed Mel filter banks are used, the caller has to ensure that they
    have correct shape.
    
    Parameters
    ----------
    power_spectrum: numpy.ndarray
        The power spectrogram from which a Mel spectrogram should be computed
    mel_fbank: numpy.ndarray, optional
        Precomputed Mel filter banks
    fs: int
        Sampling frequency of the signal from which the power spectrogram was computed. Ignored if precomputed Mel
        filter banks are used.
    window_width: int
        Window width in samples that was used to comput the power spectrogram. Ignored if precomputed Mel filter banks 
        are used.
    n_filt: int
        Number of Mel filter banks to use. Ignored if precomputed Mel filter banks are used.

    Returns
    -------
    numpy.ndarray
        Mel spectrogram computed from the specified power spectrogram
    """
    if mel_fbank is None:
        _, mel_fbank = mel_filter_bank(fs, window_width, n_filt)

    filter_banks = np.dot(mel_fbank, power_spectrum)

    return filter_banks


def power_to_db(spectrum: np.ndarray,
                clip_below: float = None,
                clip_above: float = None) -> np.ndarray:
    """
    Convert a spectrogram to the Decibel scale.
    
    Optionally, frequencies with amplitudes below or above a certain threshold can be clipped.
    
    Parameters
    ----------
    spectrum: numpy.ndarray
        The spectrogram to convert
    clip_below: float, optional
        Clip frequencies below the specified amplitude in dB
    clip_above: float, optional
        Clip frequencies above the specified amplitude in dB

    Returns
    -------
    numpy.ndarray
        The spectrogram on the Decibel scale
    """
    # there might be zeros, fix them to the lowest non-zero power in the spectrogram
    epsilon = np.min(spectrum[np.where(spectrum > 0)])

    sxx = np.where(spectrum > epsilon, spectrum, epsilon)
    sxx = 10 * np.log10(sxx / np.max(sxx))

    if clip_below is not None:
        sxx = np.maximum(sxx, clip_below)

    if clip_above is not None:
        sxx = np.minimum(sxx, clip_above)

    return sxx
