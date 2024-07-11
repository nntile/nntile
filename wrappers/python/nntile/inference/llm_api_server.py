from dataclasses import dataclass
from typing import Annotated

from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

from nntile.inference.api_server_base import (SimpleApiServerBase,
                                              SimpleApiServerParams)
from nntile.model.generation.llm import GenerationMode, GenerationParams


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


class SimpleLlmApiServer(SimpleApiServerBase):
    def __init__(self, llm_engine, params: SimpleLlmApiServerParams):
        self.llm_engine = llm_engine
        self.params = params

        super().__init__(self.params)

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
            # TODO: add logging
            generated_text = self.llm_engine.generate(
                request.text,
                params=GenerationParams(max_tokens=request.max_tokens),
                mode=request.mode,
            )
            return generated_text

        return app
