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
    test_signal_path: str
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
    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    @pytest.mark.parametrize("test_config", 
        ["test_polyphase_no_overlap_config", "test_polyphase_overlap_config"], 
        indirect=True
    )
    def test_polyphase(self, test_config: TestConfig, device):
        input_signal = np.fromfile(test_config.input_signal_path, dtype=np.complex64)
        
        test_config.module_config.use_gpu = (device == "gpu")

        if device == "gpu":
            input_signal = cp.asarray(input_signal)
        module = test_config.module_config.create_logical_instance()
        module.initialize()
        
        calc_output_raw = module.run(input_signal)  
        calc_output = calc_output_raw.get() if hasattr(calc_output_raw, 'get') else calc_output_raw
        
        num_channels = module.num_channels
        
        test_output_raw = np.fromfile(test_config.test_signal_path, dtype=np.complex64) 
        test_output = test_output_raw.reshape(-1, num_channels).T 
        
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

        # Assert only on active channels to avoid noise-floor randomness
        if active_scores.size > 0:
            assert np.all(active_scores > 1 - test_config.tol), \
                f"Similarity fail on active channels! Values: {active_scores}"
        else:
            pytest.fail("No active channels detected for validation")

    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    @pytest.mark.parametrize("test_config", 
        ["test_polyphase_no_overlap_config", "test_polyphase_overlap_config"], 
        indirect=True
    )
    def test_streaming_continuity(self, test_config: TestConfig, device):
        """
        Verify that processing a signal in chunks produces the same result 
        as processing it in one single call. This validates Buffer continuity.
        """
        input_signal = np.fromfile(test_config.input_signal_path, dtype=np.complex64)
        test_config.module_config.use_gpu = (device == "gpu")
        if device == "gpu":
            input_signal = cp.asarray(input_signal)

        module_ref = test_config.module_config.create_logical_instance()
        module_ref.initialize()
        full_output = module_ref.run(input_signal)
        full_output_np = full_output.get() if hasattr(full_output, 'get') else full_output

        module_stream = test_config.module_config.create_logical_instance()
        module_stream.initialize()
        
        chunks = np.array_split(input_signal, 3)
        chunk_outputs = []
        
        for chunk in chunks:
            chunk_res = module_stream.run(chunk)
            # Only append if the chunk was large enough to produce output
            if chunk_res.size > 0:
                chunk_outputs.append(chunk_res.get() if hasattr(chunk_res, 'get') else chunk_res)
                stream_output_np = np.concatenate(chunk_outputs, axis=1)

        min_time = min(full_output_np.shape[1], stream_output_np.shape[1])
        full_trimmed = full_output_np[:, :min_time]
        stream_trimmed = stream_output_np[:, :min_time]

        continuity_similarity = cosine_similarity(full_trimmed, stream_trimmed)
        
        print(f"Continuity Similarity Scores (per channel): {continuity_similarity}")
        
        assert np.all(continuity_similarity > 1 - test_config.tol), \
            f"Streaming continuity failed! Chunks don't match full run. Scores: {continuity_similarity}"