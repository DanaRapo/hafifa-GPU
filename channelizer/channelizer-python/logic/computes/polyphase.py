import numpy as np
import cupy as cp
import pydantic
from scipy.signal import fftconvolve

from logic.base_class import BaseClass
from utils.types import ArrayLike

class PolyphaseChannelizer(BaseClass):
    class Config(BaseClass.Config):
        channel_bw_hz: pydantic.PositiveFloat
        fs_hz: pydantic.PositiveFloat
        up_sample_factor: pydantic.PositiveInt = 1
        filter_path: str
        decimation_factor: pydantic.PositiveInt

        def create_logical_instance(self):
            return PolyphaseChannelizer(config=self)

    def initialize(self):
        self.xp = np
        self.num_channels = int(self.config.fs_hz / self.config.channel_bw_hz) # M
 
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
        itemsize = data.itemsize
        if self.config.decimation_factor ==  self.num_channels:
            num_blocks = len(data) //  self.num_channels
            reshaped = data[:num_blocks *  self.num_channels].reshape(num_blocks, self.num_channels).T
        else:
            num_blocks = (len(data) -  self.num_channels) // self.config.decimation_factor + 1
            reshaped = self.xp.lib.stride_tricks.as_strided(
                data, 
                shape=( self.num_channels, num_blocks), 
                strides=(itemsize, self.config.decimation_factor * itemsize)
            )
        x_in = self.xp.flipud(reshaped)
        if self.xp == cp:
            from cupyx.scipy.signal import fftconvolve as convolve_func
        else:
            from scipy.signal import fftconvolve as convolve_func
  
        filtered = convolve_func(x_in, current_filters, mode='same', axes=1)
        return filtered.astype(self.xp.complex64)

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