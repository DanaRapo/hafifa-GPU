import numpy as np
import cupy as cp
import pydantic

from logic.base_class import BaseClass
from utils.types import ArrayLike

class PolyphaseChannelizer(BaseClass):
    class Config(BaseClass.Config):
        channel_bw_hz: pydantic.PositiveFloat #effecting the filter
        fs_hz: pydantic.PositiveFloat
        filter_path: str
        decimation_factor: pydantic.PositiveInt
        grid_spacing_hz: pydantic.PositiveFloat

        def create_logical_instance(self):
            return PolyphaseChannelizer(config=self)

    def initialize(self):
        self.xp = np
        self.requested_channels_num = int(self.config.fs_hz / self.config.grid_spacing_hz) 
        self.num_channels = ((self.requested_channels_num + self.config.decimation_factor -1) 
                             // self.config.decimation_factor ) * self.config.decimation_factor
        self.overlap_factor = self.num_channels // self.config.decimation_factor

        filter_raw = np.fromfile(self.config.filter_path, dtype='<f4')
        filter_channel_len = len(filter_raw) // self.num_channels
    
        self.polyphase_filters = filter_raw[:self.num_channels * filter_channel_len].reshape(
            self.num_channels, filter_channel_len, order='F'
        ).astype(np.complex64)        
        
    def apply_decimation_polyphase_filter(self, current_filters:ArrayLike ,data: ArrayLike) -> ArrayLike:
        """
        Decimate the input with shifts by reshaping the input data and applys filter on channels.
        :param data: the input signal to be decimated
        :return: A matrix of the data reshaped and after polyphase filtering
        """
        total_path = self.num_channels * self.overlap_factor #M* M/R
        num_blocks = (len(data) - total_path) // self.config.decimation_factor + 1
        itemsize = data.itemsize
        reshaped = self.xp.lib.stride_tricks.as_strided(
            data,
            shape = (total_path , num_blocks),
            strides = (itemsize , self.config.decimation_factor * itemsize)
        )
        expanded_filters = self.xp.repeat(current_filters, self.overlap_factor, axis=0)
        x_in = self.xp.flipud(reshaped)
        if self.xp == cp:
            from cupyx.scipy.signal import fftconvolve as convolve_func
        else:
            from scipy.signal import fftconvolve as convolve_func
  
        filtered = convolve_func(x_in, expanded_filters, mode='same', axes=1)
        summed_data = filtered.reshape(self.num_channels, self.overlap_factor, num_blocks).sum(axis=1)
        return summed_data.astype(self.xp.complex64)

    def idft_and_freq_shift(self, data: ArrayLike) -> ArrayLike:
        """
        Apply the inverse FFT to the filtered data and frequency shift to the channelized signals 
        to center them around their respective frequencies.
        :param data: the channelized signals after filter
        :return: the frequency-shifted channelized signals
        """
        idft_data = self.xp.fft.ifft(data, axis=0) * self.num_channels
        
        num_samples = data.shape[1]
        channel_idx = self.xp.arange(self.num_channels).reshape(-1, 1)
        n = self.xp.arange(num_samples).reshape(1, -1)
        
        phase_increment = (2 * self.xp.pi * self.config.decimation_factor / self.num_channels) * channel_idx * n 
        phase_mat = self.xp.exp(-1j * phase_increment)
        return idft_data * phase_mat
        
    def run(self, data: ArrayLike) -> ArrayLike:
        """
        Execute the full polyphase channelization process.
        :param data: the input signal to be channelized, expected to be a 1D array
        :return: a matrix of channelized signals (Channels x Time)
        """
        self.xp = cp if isinstance(data, cp.ndarray) else np
        current_filters = self.xp.asarray(self.polyphase_filters)
        filtered_data = self.apply_decimation_polyphase_filter(current_filters, data)
        channels = self.idft_and_freq_shift(filtered_data)
        
        return channels