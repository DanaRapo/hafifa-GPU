import abc

import numpy as np
from pydantic import BaseModel

from utils.types import ArrayLike

class BaseClass(abc.ABC):
    class Config(BaseModel):
        """Base config for all computing logics"""

        def create_logical_instance(self) -> "BaseClass":
            pass

    def __init__(self, config: "BaseClass.Config"):
        self.config = config
        self.initialize()

    def initialize(self):
        pass

    @abc.abstractmethod
    def run(self, data: ArrayLike) -> ArrayLike:
        pass