import numpy as np
import cupy as cp
import pydantic
import pytest
import hydra
from omegaconf import OmegaConf

from logic.computes.polyphase import PolyphaseChannelizer
from utils.energy_similarity import energy_similarity


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
        if device == "gpu":
            input_signal = cp.asarray(input_signal)
        module = test_config.module_config.create_logical_instance()
        module.initialize()
        calc_output_raw = module.run(input_signal)  
        calc_output = calc_output_raw.get() if hasattr(calc_output_raw, 'get') else calc_output_raw
        num_channels = int(test_config.module_config.fs_hz // test_config.module_config.grid_spacing_hz)
        test_output_raw = np.fromfile(test_config.test_signal_path, dtype=np.complex64) 
        test_output = test_output_raw.reshape(-1, num_channels).T 
        energy_ratios = energy_similarity(calc_output, test_output)
        print(f"Energy Ratios (Python/Matlab): {energy_ratios}")
        diffs = np.abs(energy_ratios - 1.0)
        assert np.all(diffs < test_config.tol), f"Energy mismatch on {device}! Ratios: {energy_ratios}"