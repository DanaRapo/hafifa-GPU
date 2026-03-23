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
        overlap_factor: pydantic.PositiveInt
        use_gpu: bool = False

        def create_logical_instance(self):
            return PolyphaseChannelizer(config=self)

    def initialize(self):
        self.xp = cp if self.config.use_gpu else np
        self.requested_channels_num = int(self.config.fs_hz / self.config.channel_bw_hz) 
        self.decimation_factor = self.requested_channels_num // self.config.overlap_factor

        self.num_channels = int(self.requested_channels_num * self.config.overlap_factor) 
        filter_raw = np.fromfile(self.config.filter_path, dtype='<f4')
        
        if filter_raw.size % self.num_channels != 0:
            raise ValueError(
                f"Filter size {filter_raw.size} is not divisible by "
                f"num_channels {self.num_channels}. Check filter design parameters."
            )
        self.ola_param = len(filter_raw) // self.num_channels
        try:
            polyphase_filters_np = filter_raw.reshape(
                self.num_channels, self.ola_param, order='F'
            ).astype(np.complex64)   
            self.polyphase_filters = self.xp.asarray(polyphase_filters_np)
        
        except ValueError as e:
            raise ValueError(f"Reshape failed despite size check: {e}")

        self.prev_buffer = None
        self.total_processed_blocks = 0     
        
    def apply_decimation_polyphase_filter(self ,data: ArrayLike) -> ArrayLike:
        """
        Decimate the input with shifts by reshaping the input data and applys filter on channels.
        :param data: the input signal to be decimated
        :return: A matrix of the data reshaped and after polyphase filtering
        """
        num_blocks = (len(data) - self.num_channels * self.ola_param) // self.decimation_factor + 1
        itemsize = data.itemsize
       
        reshaped_data = self.xp.lib.stride_tricks.as_strided(
            data,
            shape=(self.ola_param, self.num_channels, num_blocks),
            strides=(self.num_channels * itemsize, itemsize, self.decimation_factor * itemsize)
        )
        expanded_filters = self.polyphase_filters.T.reshape(self.ola_param, self.num_channels, 1)
        data_in = reshaped_data[::-1, ::-1, :]
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
        n_global = self.xp.arange(self.total_processed_blocks, 
                                  self.total_processed_blocks + num_samples).reshape(1, -1)
        
        phase_increment = (2 * self.xp.pi * self.decimation_factor / self.num_channels) * channel_idx * n_global 
        phase_mat = self.xp.exp(-1j * phase_increment)
        
        self.total_processed_blocks += num_samples
        return idft_data * phase_mat
        
    def run(self, data: ArrayLike) -> ArrayLike:
        """
        Execute the full polyphase channelization process.
        :param data: the input signal to be channelized, expected to be a 1D array
        :return: a matrix of channelized signals (Channels x Time)
        """
        if self.prev_buffer is None:
            full_data = data
        else:
            full_data = self.xp.concatenate([self.prev_buffer, data])

        if len(full_data) < (self.num_channels * self.ola_param) :
            self.prev_buffer = full_data
            return self.xp.array([], dtype=self.xp.complex64)
        
        num_blocks = (len(full_data) - self.num_channels * self.ola_param) // self.decimation_factor + 1
        last_idx_to_process = (num_blocks - 1) * self.decimation_factor + self.num_channels * self.ola_param

        data_to_process = full_data[:last_idx_to_process]
        self.prev_buffer = full_data[num_blocks * self.decimation_factor:]

        filtered_data = self.apply_decimation_polyphase_filter(data_to_process)
        channels = self.idft_and_freq_shift(filtered_data)
        
        return channels