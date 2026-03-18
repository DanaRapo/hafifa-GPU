import numpy as np
import cupy as cp
import pydantic
from typing import Any

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
        self.num_channels = int(self.config.fs_hz / self.config.channel_bw_hz) #M
        filter_data = np.fromfile(self.config.filter_path, dtype=np.complex64)
        filter = self.xp.array(filter_data)
        filter_len = len(filter) // self.num_channels
        self.polyphase_filters = filter.reshape(filter_len, self.num_channels).T
    
     def apply_decimation_polyphase_filter(self, xp: Any, data: ArrayLike) -> ArrayLike:
        """
        Decimate the input with shifts by reshaping the input data and applys filter on channels.
        :param data: the input signal to be decimated
        :return: A metrix of the data reshaped and after polyphase filtering
        """
        if self.config.decimation_factor == self.num_channels:
            reshaped = data.reshape(self.num_channels, -1, order='F')
        else:
            num_blocks = (len(data) - self.num_channels) // self.config.decimation_factor + 1
            # Organize the data right when there is overlap 
            itemsize = data.itemsize
            reshaped = self.xp.lib.stride_tricks.as_strided(
                data, 
                shape=(self.num_channels, num_blocks), 
                strides=(itemsize, self.config.decimation_factor * itemsize)
            )
        if xp == cp:
            from cupyx.scipy.signal import convolve
        else:
            from scipy.signal import convolve
        filtered = convolve(reshaped, self.polyphase_filters, mode='same', method='direct')
        return filtered
    
     def idft_and_freq_shift(self, xp: Any, data: ArrayLike)-> ArrayLike:
        """
         Apply the inverse FFT to the filtered data and frequency shift to the channelized signals to center them
           around their respective frequencies.
        :param data: the channelized signals after filter
        :return: the frequency-shifted channelized signals
        """
        idft_data = xp.fft.ifft(data, axis=0)
        num_samples = data.shape[1]
        k = xp.arange(self.num_channels).reshape(-1, 1)
        n = xp.arange(num_samples).reshape(1, -1)
        phase_increament = (2 * xp.pi * self.config.decimation_factor / self.num_channels) * k * n 
        phase_mat = xp.exp(-1j * phase_increament)
        shifted_data = idft_data * phase_mat
        return shifted_data
       
     def run(self, data: ArrayLike) -> ArrayLike:
        """
        :param data: the input signal to be channelized, expected to be a 1D numpy array representing the time-domain signal
        :return: a list of channelized signals
        """
        if isinstance(data, cp.ndarray):
            xp = cp
        else:
            xp = np
        filterd_data = self.apply_decimation_polyphase_filter(xp, data)
        channels = self.idft_and_freq_shift(xp, filterd_data)
        return channels