import numpy as np
import pydantic
from scipy.signal import resample_poly

from Solution.logic.Base_Class import BaseClass


class BasicChannelizer(BaseClass):
    class Config(BaseClass.Config):
        channel_bw_hz: pydantic.PositiveFloat
        fs_hz: pydantic.PositiveFloat
        up_sample_factor: pydantic.PositiveInt = 1

        def create_logical_instance(self):
            return BasicChannelizer(config=self)
        
    def initialize(self):
        self.num_channels = int(self.config.fs_hz / self.config.channel_bw_hz)

    def frequency_shift(self, data: np.ndarray, freq_shift: float) -> np.ndarray:
        """

        :param data: the input signal to be frequency shifted
        :param freq_shift: the frequency shift in Hz
        :return: the frequency-shifted signal
        """
        n = len(data)
        phase_increment = -2 * np.pi * freq_shift / self.config.fs_hz
        phases = phase_increment * np.arange(n)
        return data * np.exp(1j * phases)

    def lpf_Decimation(self, data: np.ndarray) -> np.ndarray:
        """

        :param data: the input signal to be low-pass filtered and decimated
        :return: the low-pass filtered and decimated signal
        """
        return resample_poly(data,
                             up=self.config.up_sample_factor,
                             down=self.num_channels)

    def run(self, data: np.ndarray) -> np.ndarray:
        """

        :param data: the input signal to be channelized, expected to be a 1D numpy array representing the time-domain signal
        :return: a list of channelized signals
        """
        channels = []
        # Calculate frequencies from -fs/2 to +fs/2
        cyclic_frequencies = np.fft.fftfreq(self.num_channels, 1/self.config.fs_hz)
        for freq in cyclic_frequencies:
            shifted_data = self.frequency_shift(data, freq)
            filtered_decimated_data = self.lpf_Decimation(shifted_data)
            channels.append(filtered_decimated_data)

        return np.array(channels)