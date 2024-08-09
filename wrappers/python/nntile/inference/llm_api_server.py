# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/inference/llm_api_server.py
#
# @version 1.1.0

import logging
from dataclasses import dataclass
from typing import Annotated

from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

from nntile.inference.api_server_base import (
    SimpleApiServerBase, SimpleApiServerParams)
from nntile.model.generation.llm import GenerationMode, GenerationParams

logger = logging.getLogger(__name__)


@dataclass
class SimpleLlmApiServerParams(SimpleApiServerParams):
    pass


class SimpleLlmApiServerGenerateRequest(BaseModel):
    text: str
    max_tokens: int = Field(
        default=0,
        description="How much tokens to generate including initial text",
    )
    mode: GenerationMode = Field(
        default=GenerationMode.Greedy, description="Strategy for generation"
    )
    use_cache: bool = Field(
        default=True, description="Use key value cache for generation"
    )
    need_static_padding: bool = Field(
        default=False,
        description="Padd input tensor and use static model via forward_async",
    )


class SimpleLlmApiServer(SimpleApiServerBase):
    def __init__(self, llm_engine, params: SimpleLlmApiServerParams):
        super().__init__(params)
        self.llm_engine = llm_engine

    def get_app(self):
        app = FastAPI()

        @app.get("/info")
        def info():
            return "I am gpt2 model!"

        @app.post("/generate")
        def generate(
            request: Annotated[
                SimpleLlmApiServerGenerateRequest, Body(embed=True)
            ],
        ):
            logger.info(
                f"Start generating for request: {request.model_dump()}"
            )
            generated_text = self.llm_engine.generate(
                request.text,
                params=GenerationParams(
                    max_tokens=request.max_tokens,
                    use_cache=request.use_cache,
                    need_static_padding=request.need_static_padding,
                ),
                mode=request.mode,
            )
            return generated_text

        return app
