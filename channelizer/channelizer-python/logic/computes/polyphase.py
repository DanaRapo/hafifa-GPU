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
        use_gpu: bool = False

        def create_logical_instance(self):
            return PolyphaseChannelizer(config=self)

    def initialize(self):
        self.xp = cp if self.config.use_gpu else np
        self.requested_channels_num = int(self.config.fs_hz / self.config.channel_bw_hz) 
        self.overlap_factor = self.requested_channels_num // self.config.decimation_factor

        self.num_channels = int(self.requested_channels_num * self.overlap_factor) 
        filter_raw = np.fromfile(self.config.filter_path, dtype='<f4')
        
        assert len(filter_raw) % self.num_channels == 0, \
            f"Filter length {len(filter_raw)} must be divisible by {self.num_channels}"
        
        self.ola_param = len(filter_raw) // self.num_channels
    
        polyphase_filters_np = filter_raw.reshape(
            self.num_channels, self.ola_param, order='F'
        ).astype(np.complex64)   
        self.polyphase_filters = self.xp.asarray(polyphase_filters_np)     
        
    def apply_decimation_polyphase_filter(self ,data: ArrayLike) -> ArrayLike:
        """
        Decimate the input with shifts by reshaping the input data and applys filter on channels.
        :param data: the input signal to be decimated
        :return: A matrix of the data reshaped and after polyphase filtering
        """
        num_blocks = (len(data) - self.num_channels * self.ola_param) // self.config.decimation_factor + 1
        itemsize = data.itemsize
       
        reshaped = self.xp.lib.stride_tricks.as_strided(
            data,
            shape=(self.ola_param, self.num_channels, num_blocks),
            strides=(self.num_channels * itemsize, itemsize, self.config.decimation_factor * itemsize)
        )
        expanded_filters = self.polyphase_filters.T.reshape(self.ola_param, self.num_channels, 1)
        data_in = reshaped[::-1, ::-1, :]
        filtered = data_in * expanded_filters
        summed_data = filtered.sum(axis=0)
       
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
        
        filtered_data = self.apply_decimation_polyphase_filter(data)
        channels = self.idft_and_freq_shift(filtered_data)
        
        return channels