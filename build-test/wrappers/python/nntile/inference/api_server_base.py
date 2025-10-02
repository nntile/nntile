# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/inference/api_server_base.py
#
# @version 1.1.0

from abc import ABC, abstractmethod
from dataclasses import dataclass

import uvicorn


@dataclass
class SimpleApiServerParams:
    host: str = "127.0.0.1"
    port: int = 12224


class SimpleApiServerBase(ABC):
    def __init__(self, params: SimpleApiServerParams):
        self.app = None
        self.params = params

    @abstractmethod
    def get_app(self):
        pass

    def run(self):
        self.app = self.get_app()
        uvicorn.run(self.app, host=self.params.host, port=self.params.port)
