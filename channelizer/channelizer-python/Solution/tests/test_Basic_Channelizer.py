import numpy as np
import pydantic
import pytest
import hydra
from omegaconf import OmegaConf

from Solution.logic.computes.Basic_Channelizer import BasicChannelizer
from Solution.utils.cosine_similarity import cosine_similarity


class TestConfig(pydantic.BaseModel):
    module_config: BasicChannelizer.Config
    input_signal_path: str
    test_signal_path: str
    tol: pydantic.PositiveFloat = 3e-2


@pytest.fixture(scope="session")
def test_config(request) -> TestConfig:
    scenario_name = request.param
    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(config_name=scenario_name)
    test_cfg_dict = OmegaConf.to_container(cfg.test_config, resolve=True)
    test_cfg = TestConfig.model_validate(test_cfg_dict)
    return test_cfg


class TestClass:
    @pytest.mark.parametrize("test_config", ["test_basic_channelizer_config"], indirect=True)
    def test_basic_channelizer(self, test_config: TestConfig):
        input_signal = np.fromfile(test_config.input_signal_path, dtype=np.complex64)
        module = test_config.module_config.create_logical_instance()
        calc_output = module.run(input_signal)
        num_channels = int(test_config.module_config.fs_hz // test_config.module_config.channel_bw_hz)
        test_output_raw = np.fromfile(test_config.test_signal_path, dtype=np.complex64) 
        test_output = test_output_raw.reshape(num_channels, -1) 
        calc_output = np.array(calc_output)
        assert calc_output.shape == test_output.shape
        mag_calc = np.abs(calc_output)
        mag_test = np.abs(test_output)
        similarities = cosine_similarity(mag_calc, mag_test)
        print(f"Similarities : {similarities}")
        assert np.all(similarities > 1 - test_config.tol), f"Similarity fail! Values: {similarities}"