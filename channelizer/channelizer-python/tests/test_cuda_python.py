import numpy as np
import cupy as cp
import pydantic
import pytest
import hydra
from omegaconf import OmegaConf

from logic.computes.polyphase import PolyphaseChannelizer
from utils.cosine_similarity import cosine_similarity


class TestConfig(pydantic.BaseModel):
    module_config: PolyphaseChannelizer.Config
    input_signal_path: str
    cuda_signal_path: str
    tol: pydantic.PositiveFloat = 9e-2


@pytest.fixture(scope="session")
def test_config(request) -> TestConfig:
    scenario_name = request.param
    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(config_name=scenario_name)
    test_cfg_dict = OmegaConf.to_container(cfg.test_config, resolve=True)
    test_cfg = TestConfig.model_validate(test_cfg_dict)
    return test_cfg


class TestClass:
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("test_config", 
        ["test_cublas_noOverlap_config"], 
        indirect=True
    )
    def test_polyphase_cublas(self, test_config: TestConfig, device):
        input_signal = np.fromfile(test_config.input_signal_path, dtype=np.complex64)
        
        test_config.module_config.use_gpu = (device == "gpu")

        if device == "gpu":
            input_signal = cp.asarray(input_signal)
        module = test_config.module_config.create_logical_instance()
        module.initialize()
        
        calc_output_raw = module.apply_decimation_polyphase_filter(input_signal)  
        calc_output = calc_output_raw.get() if hasattr(calc_output_raw, 'get') else calc_output_raw
        
        num_channels = module.num_channels
        
        cuda_output_raw = np.fromfile(test_config.cuda_signal_path, dtype=np.complex64) 
        test_output = cuda_output_raw.reshape(-1, num_channels).T
        min_samples = min(calc_output.shape[1], test_output.shape[1])
        calc_output_trimmed = calc_output[:, :min_samples]
        test_output_trimmed = test_output[:, :min_samples]
            
        similarity_scores = cosine_similarity(calc_output_trimmed, test_output_trimmed)
        print(f"Energy Ratios (Python/Matlab): {similarity_scores}")
        
        energies = np.mean(np.abs(calc_output_trimmed)**2, axis=1)
        relative_energies = energies / (np.max(energies) + 1e-12)
        
        significant_channel_mask = relative_energies > 0.01 
        active_scores = similarity_scores[significant_channel_mask]
        
        print(f"Energy Ratios (Python/Matlab): {similarity_scores}")
        print(f"Significant Channels Mask: {significant_channel_mask}")
        print(f"DEBUG: calc_output shape: {calc_output.shape}")
        print(f"DEBUG: test_output shape: {test_output.shape}")
        print(f"DEBUG: calc_output samples (first 5): {calc_output[0, :5]}")
        print(f"DEBUG: test_output samples (first 5): {test_output[0, :5]}")

        
        test_output_flipped = test_output_trimmed[::-1, :]

        similarity_scores_flipped = cosine_similarity(calc_output_trimmed, test_output_flipped)
        print(f"Flipped Similarity: {similarity_scores_flipped}")

        final_calc_output = module.run(input_signal)
        cuda_res_final = module.idft_and_freq_shift(test_output)
        if hasattr(cuda_res_final, 'get'): cuda_res_final = cuda_res_final.get()
        final_similarity = cosine_similarity(final_calc_output, cuda_res_final[:, :min_samples])

        print(f"Final Channelizer Similarity (After Phase Correction): {final_similarity}")
        # Assert only on active channels to avoid noise-floor randomness
        if active_scores.size > 0:
            assert np.all(active_scores > 1 - test_config.tol), \
                f"Similarity fail on active channels! Values: {active_scores}"
        else:
            pytest.fail("No active channels detected for validation")


    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("test_config", 
        ["test_cublas_overlap_config"], 
        indirect=True
    )
    def test_polyphase_cublas_overlap(self, test_config: TestConfig, device):
        input_signal = np.fromfile(test_config.input_signal_path, dtype=np.complex64)
        test_config.module_config.use_gpu = (device == "gpu")

        if device == "gpu":
            input_signal = cp.asarray(input_signal)
        
        module = test_config.module_config.create_logical_instance()
        module.initialize()

        calc_output_raw = module.apply_decimation_polyphase_filter(input_signal)
        calc_output = calc_output_raw.get() if hasattr(calc_output_raw, 'get') else calc_output_raw

        num_channels = module.num_channels
        cuda_output_raw = np.fromfile(test_config.cuda_signal_path, dtype=np.complex64)

        test_output = cuda_output_raw.reshape((-1, num_channels)).T

        min_samples = min(calc_output.shape[1], test_output.shape[1])
        calc_output_trimmed = calc_output[:, :min_samples]
        test_output_trimmed = test_output[:, :min_samples]

        similarity_scores = cosine_similarity(calc_output_trimmed, test_output_trimmed)
        print(f"Energy Ratios : {similarity_scores}")
        
        energies = np.mean(np.abs(calc_output_trimmed)**2, axis=1)
        relative_energies = energies / (np.max(energies) + 1e-12)
        
        significant_channel_mask = relative_energies > 0.01 
        active_scores = similarity_scores[significant_channel_mask]
        
        print(f"Energy Ratios no overlap: {similarity_scores}")
        print(f"Significant Channels Mask: {significant_channel_mask}")

        # Assert only on active channels to avoid noise-floor randomness
        if active_scores.size > 0:
            assert np.all(active_scores > 1 - test_config.tol), \
                f"Similarity fail on active channels! Values: {active_scores}"
        else:
            pytest.fail("No active channels detected for validation")

    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("test_config", 
        ["test_cuda_python_noOverlap_config"], 
        indirect=True
    )
    def test_polyphase_direct_no_overlap(self, test_config: TestConfig, device):
        input_signal = np.fromfile(test_config.input_signal_path, dtype=np.complex64)
        test_config.module_config.use_gpu = (device == "gpu")

        if device == "gpu":
            input_signal = cp.asarray(input_signal)
        
        module = test_config.module_config.create_logical_instance()
        module.initialize()

        calc_output_raw = module.apply_decimation_polyphase_filter(input_signal)
        calc_output = calc_output_raw.get() if hasattr(calc_output_raw, 'get') else calc_output_raw

        num_channels = module.num_channels
        cuda_output_raw = np.fromfile(test_config.cuda_signal_path, dtype=np.complex64)

        test_output = cuda_output_raw.reshape((-1, num_channels)).T

        min_samples = min(calc_output.shape[1], test_output.shape[1])
        calc_output_trimmed = calc_output[:, :min_samples]
        test_output_trimmed = test_output[:, :min_samples]

        similarity_scores = cosine_similarity(calc_output_trimmed, test_output_trimmed)
        print(f"Energy Ratios : {similarity_scores}")
        
        energies = np.mean(np.abs(calc_output_trimmed)**2, axis=1)
        relative_energies = energies / (np.max(energies) + 1e-12)
        
        significant_channel_mask = relative_energies > 0.01 
        active_scores = similarity_scores[significant_channel_mask]
        
        print(f"Energy Ratios no overlap: {similarity_scores}")
        print(f"Significant Channels Mask: {significant_channel_mask}")

        # Assert only on active channels to avoid noise-floor randomness
        if active_scores.size > 0:
            assert np.all(active_scores > 1 - test_config.tol), \
                f"Similarity fail on active channels! Values: {active_scores}"
        else:
            pytest.fail("No active channels detected for validation")

    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("test_config", 
        ["test_cuda_python_overlap_config"], 
        indirect=True
    )
    def test_polyphase_direct_overlap(self, test_config: TestConfig, device):
        input_signal = np.fromfile(test_config.input_signal_path, dtype=np.complex64)
        test_config.module_config.use_gpu = (device == "gpu")

        if device == "gpu":
            input_signal = cp.asarray(input_signal)
        
        module = test_config.module_config.create_logical_instance()
        module.initialize()

        calc_output_raw = module.apply_decimation_polyphase_filter(input_signal)
        calc_output = calc_output_raw.get() if hasattr(calc_output_raw, 'get') else calc_output_raw

        num_channels = module.num_channels
        cuda_output_raw = np.fromfile(test_config.cuda_signal_path, dtype=np.complex64)

        test_output = cuda_output_raw.reshape((-1, num_channels)).T

        min_samples = min(calc_output.shape[1], test_output.shape[1])
        calc_output_trimmed = calc_output[:, :min_samples]
        test_output_trimmed = test_output[:, :min_samples]

        similarity_scores = cosine_similarity(calc_output_trimmed, test_output_trimmed)
        print(f"Energy Ratios overlap: {similarity_scores}")
        
        test_output_flipped = test_output_trimmed[::-1, :]

        similarity_scores_flipped = cosine_similarity(calc_output_trimmed, test_output_flipped)
        print(f"Flipped Similarity: {similarity_scores_flipped}")

        energies = np.mean(np.abs(calc_output_trimmed)**2, axis=1)
        relative_energies = energies / (np.max(energies) + 1e-12)
        
        significant_channel_mask = relative_energies > 0.01 
        active_scores = similarity_scores[significant_channel_mask]
        
        print(f"Energy Ratios: {similarity_scores}")
        print(f"Significant Channels Mask: {significant_channel_mask}")

        # Assert only on active channels to avoid noise-floor randomness
        if active_scores.size > 0:
            assert np.all(active_scores > 1 - test_config.tol), \
                f"Similarity fail on active channels! Values: {active_scores}"
        else:
            pytest.fail("No active channels detected for validation")