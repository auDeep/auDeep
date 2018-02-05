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

"""Data set preprocessor for spectrogram extraction"""
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import soundfile

from audeep.backend.log import LoggingMixin
from audeep.backend.signal import spectral


class ChannelFusion(Enum):
    """
    Channel fusion strategies in case there is more than one channel in the input audio.
    """
    MEAN = 0
    DIFF = 1
    LEFT = 2
    RIGHT = 3


class Preprocessor(LoggingMixin):
    """
    A data set preprocessor which extracts spectrograms from raw audio files.
    """

    def __init__(self,
                 channel_fusion: ChannelFusion,
                 pre_emphasis: bool,
                 window_width: float,
                 window_overlap: float,
                 mel_bands: int = None,
                 clip_power_above: int = None,
                 clip_power_below: int = None,
                 chunk_length: float = None,
                 chunk_count: int = None,
                 fixed_length: float = None,
                 center_fixed: bool = False,
                 mean_norm: bool = True):
        """
        Create and initialize a Preprocessor with the specified parameters.
        
        Parameters
        ----------
        channel_fusion: ChannelFusion
            Channel fusion strategy specifying how audio data with multiple channels should be handled
        pre_emphasis: bool
            Indicates whether to apply a pre-emphasis filter to the raw audio
        window_width: float
            Width in seconds of the FFT windows used to extract spectrograms
        window_overlap: float
            Overlap in seconds between FFT windows
        mel_bands: int, optional
            If set, compress the spectrograms by computing the specified number of Mel frequency bands
        clip_power_above: int, optional
            Clip amplitudes above the specified dB value
        clip_power_below: int, optional
            Clip amplitudes below the specified dB value
        chunk_length: float, optional
            Split the raw audio into chunks of the specified length in seconds
        chunk_count: int, optional
            Split the audio into the specified number of chunks with length `chunk_length`. Excess chunks will be 
            discarded. 
        fixed_length: float, optional
            Pad or cut raw audio to have the specified length in seconds. Useful to ensure that chunking works as
            expected
        center_fixed: bool, optional
            By default, the first `fixed_length` seconds of audio are used if the `fixed_length` parameter is given.
            If `center_fixed` is set, the fixed length window is centered over the raw audio.
        mean_norm: bool, optional
            Apply mean normalization to spectrograms
        """
        super().__init__()

        self._channel_fusion = channel_fusion
        self._pre_emphasis = pre_emphasis
        self._window_width = window_width
        self._window_overlap = window_overlap
        self._mel_bands = mel_bands
        self._clip_power_above = clip_power_above
        self._clip_power_below = clip_power_below
        self._chunk_length = chunk_length
        self._chunk_count = chunk_count
        self._mel_fbanks = {}
        self._freq_points = {}
        self._fixed_length = fixed_length
        self._center_fixed = center_fixed
        self._mean_norm = mean_norm

    def process(self,
                file: Path) -> (List[np.ndarray], np.ndarray, List[np.ndarray]):
        """
        Process the audio file at the specified path, according to the configuration of this Preprocessor.
        
        Parameters
        ----------
        file: pathlib.Path
            The audio file to process.

        Returns
        -------
        sxx: list of numpy.ndarray
            A list containing the spectrograms of each chunk, in order. Time is returned as the last axis.
        f: numpy.ndarray
            Frequency values in Hertz for each entry on the frequency axis
        t: list of numpy.ndarray
            A list containing the time values in seconds for each entry on the time axis of each chunk, in order
        """
        if not file.exists():
            raise IOError("failed to open audio file at {}".format(file))

        # STEP 0: read data
        samples, fs = soundfile.read(str(file), always_2d=True)

        if self._fixed_length is not None:
            fixed_length_samples = int(self._fixed_length * fs)

            num_samples = samples.shape[0]

            if num_samples > fixed_length_samples:
                if self._center_fixed:
                    offset = (num_samples - fixed_length_samples) // 2
                else:
                    offset = 0

                samples = samples[offset:offset + fixed_length_samples]
            else:
                if self._center_fixed:
                    pad_start = (fixed_length_samples - num_samples) // 2
                    pad_end = fixed_length_samples - num_samples - pad_start
                else:
                    pad_start = 0
                    pad_end = fixed_length_samples - num_samples

                samples = np.pad(samples, pad_width=((pad_start, pad_end), (0, 0)),
                                 mode="constant")

        self.log.debug("file %s: %d samples @ %d Hz", file, samples.shape[0], fs)

        # STEP 1: fuse channels
        if samples.shape[1] == 2:
            if self._channel_fusion == ChannelFusion.MEAN:
                samples = np.mean(samples, axis=1)
            elif self._channel_fusion == ChannelFusion.DIFF:
                samples = samples[:, 0] - samples[:, 1]
            elif self._channel_fusion == ChannelFusion.LEFT:
                samples = samples[:, 0]
            elif self._channel_fusion == ChannelFusion.RIGHT:
                samples = samples[:, 1]
            else:
                raise ValueError("unknown channel fusion strategy: {}".format(self._channel_fusion))

        # STEP 2: pre-emphasis (optional)
        if self._pre_emphasis:
            samples = spectral.pre_emphasis_filter(samples)

        # STEP 3: power spectrum
        # due to weird bugs when the window width in samples is not an integer, perform some additional checking here
        window_width_samples = int(fs * self._window_width)
        window_overlap_samples = int(fs * self._window_overlap)

        if self._fixed_length is not None:
            expected_step = self._window_width - self._window_overlap
            expected_num_frames = int((self._fixed_length - self._window_overlap) / expected_step)

            actual_step = window_width_samples - window_overlap_samples
            actual_num_frames = (samples.shape[0] - window_overlap_samples) // actual_step

            if actual_num_frames > expected_num_frames:
                self.log.warning("due to rounding errors, there are more frames than expected. decreasing window "
                                 "overlap by one sample")

                window_overlap_samples -= 1
            elif actual_num_frames < expected_num_frames:
                self.log.warning("due to rounding errors, there are fewer frames than expected. increasing window "
                                 "overlap by one sample")

                window_overlap_samples += 1

        window_step_samples = window_width_samples - window_overlap_samples

        f, t, sxx = spectral.power_spectrum(samples, fs, window_width_samples, window_overlap_samples)

        # STEP 4: convert to mel spectrum (optional)
        if self._mel_bands is not None:
            if fs not in self._mel_fbanks:
                freq_points, mel_fbank = spectral.mel_filter_bank(fs, window_width_samples,
                                                                  self._mel_bands)

                valid_banks = np.sum(mel_fbank, axis=1) > 0
                zero_count = self._mel_bands - len(np.nonzero(valid_banks)[0])

                if zero_count > 0:
                    self.log.warning("frequency resolution is not high enough for the specified number of mel filters, "
                                     "there will be %d zero bands", zero_count)

                self._freq_points[fs] = freq_points
                self._mel_fbanks[fs] = mel_fbank

            f = self._freq_points[fs]
            sxx = spectral.mel_spectrum(sxx, mel_fbank=self._mel_fbanks[fs])

        # STEP 5: convert power scale to dB scale, optionally with amplitude clipping
        sxx = spectral.power_to_db(sxx, clip_above=self._clip_power_above, clip_below=self._clip_power_below)

        # STEP 6: split into chunks (optional)
        if self._chunk_length is not None:
            chunk_length_frames = int(fs * self._chunk_length)
            chunk_length_windows = chunk_length_frames // window_step_samples
            num_windows = sxx.shape[1]

            # last axis of sxx is time
            indices = np.arange(chunk_length_windows, num_windows, chunk_length_windows)
            sxx = np.split(sxx, indices_or_sections=indices, axis=1)
            t = np.split(t, indices_or_sections=indices)

            if sxx[-1].shape[1] != sxx[0].shape[1] and len(sxx) < self._chunk_count + 1:
                raise ValueError("too few chunks: expected at least {}, got {}".format(self._chunk_count + 1, len(sxx)))

            sxx = sxx[:self._chunk_count]
            t = t[:self._chunk_count]
        else:
            sxx = [sxx]
            t = [t]

        # STEP 7: mean normalization
        if self._mean_norm:
            sxx = [x - np.mean(x) for x in sxx]
            sxx_min = [np.min(x) for x in sxx]
            sxx_max = [np.max(x) for x in sxx]

            result = []

            for x, x_min, x_max in zip(sxx, sxx_min, sxx_max):
                if abs(x_max - x_min) > 1e-4:
                    result.append(2 * (x - x_min) / (x_max - x_min) - 1)
                else:
                    result.append(x - x_min)

            return result, f, t
        else:
            return sxx, f, t
