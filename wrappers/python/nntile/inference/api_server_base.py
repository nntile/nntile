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
