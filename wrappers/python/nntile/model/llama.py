# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/llama.py
# Llama model of NNTile Python package
#
# @version 1.0.0

from typing import Dict

from nntile.layer import Act, Linear, Prod
from nntile.model.base_model import BaseModel
from nntile.tensor import TensorMoments, notrans


class LlamaConfig(Dict):
    def __init__(
        self,
        vocab_size: int = 32000,
        vocab_embed_dim_tile: int = 32000,
        hidden_size: int = 4096,
        hidden_size_tile: int = 4096,
        max_position_embeddings: int = 2048,
        intermediate_size: int = 11008,
        intermediate_size_tile: int = 11008,
        rms_norm_eps: float = 1e-06,
        num_hidden_layers: int = 32,
        n_attention_head: int = 32,
        n_head_tile: int = 32,
        num_key_value_heads: int = 32,
        num_key_value_head_tile: int = 32,
        activation_function: str = "silu",
        flashattention: bool = True,
        use_redux: bool = False,
        dtype: str = "fp32",
        eos_token_id: int = 2,
        bos_token_id: int = 1,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_theta: float = 10000.,
        mlp_bias: bool = False
    ):
        self["vocab_size"] = vocab_size
        self["vocab_embed_dim_tile"] = vocab_embed_dim_tile
        self["hidden_size"] = hidden_size
        self["hidden_size_tile"] = hidden_size_tile
        self["max_position_embeddings"] = max_position_embeddings
        self["intermediate_size"] = intermediate_size
        self["intermediate_size_tile"] = intermediate_size_tile
        self["rms_norm_eps"] = rms_norm_eps
        self["num_hidden_layers"] = num_hidden_layers
        self["n_attention_head"] = n_attention_head
        self["n_head_tile"] = n_head_tile
        self["num_key_value_heads"] = num_key_value_heads
        self["num_key_value_head_tile"] = num_key_value_head_tile
        self["activation_function"] = activation_function
        self["flashattention"] = flashattention
        self["redux"] = use_redux
        self["dtype"] = dtype
        self["eos_token_id"] = eos_token_id
        self["bos_token_id"] = bos_token_id
        self["attention_bias"] = attention_bias
        self["attention_dropout"] = attention_dropout
        self["rope_theta"] = rope_theta
        self["mlp_bias"] = mlp_bias

    def __getattr__(self, attr):
        return self[attr]


class LlamaMLP(BaseModel):
    next_tag: int

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, config: LlamaConfig, next_tag: int):
        # Init activations and list of layers
        activations = [x]
        layers = []
        hidden_size = config["hidden_size"]
        hidden_size_tile = config["hidden_size_tile"]
        intermediate_size = config["intermediate_size"]
        intermediate_size_tile = config["intermediate_size_tile"]
        activation_function = config["activation_function"]
        redux = config["redux"]
        mlp_bias = config["mlp_bias"]
        gemm_ndim = 1
        # Initial linear layer that converts input to internal shape
        gate_proj, next_tag = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [intermediate_size],
            [intermediate_size_tile],
            next_tag,
            redux=redux,
            bias=mlp_bias
        )
        layers.append(gate_proj)
        activations.extend(gate_proj.activations_output)

        new_layer, next_tag = Act.generate_simple(
            activations[-1], activation_function, next_tag
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        up_proj, next_tag = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [intermediate_size],
            [intermediate_size_tile],
            next_tag,
            redux=redux,
            bias=mlp_bias
        )
        layers.append(up_proj)
        activations.extend(up_proj.activations_output)
        self.next_tag = next_tag

        prod_layer, next_tag = Prod.generate_simple(
            activations[-2], activations[-1], next_tag
        )
        layers.append(prod_layer)
        activations.extend(prod_layer.activations_output)

        down_proj, next_tag = Linear.generate_simple(
            activations[-1],
            "R",
            notrans,
            gemm_ndim,
            [hidden_size],
            [hidden_size_tile],
            next_tag,
            redux=redux,
            bias=mlp_bias
        )
        layers.append(down_proj)
        activations.extend(down_proj.activations_output)
        self.next_tag = next_tag
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    # Randomly init all linear layers
    def init_randn_async(self):
        for layer in self.layers:
            if type(layer) is Linear:
                layer.init_randn_async()

    @staticmethod
    def from_torch(
        torch_mlp, x: TensorMoments, config: LlamaConfig, next_tag: int
    ):
        """
        torch_mlp is PyTorch MLP where no biases in linear layers
        """
        llama_ff_nntile = LlamaMLP(x, config, next_tag)
        torch_params = list(torch_mlp.parameters())
        for i, p in enumerate(llama_ff_nntile.parameters):
            p.value.from_array(torch_params[i].cpu().detach().numpy())
        return llama_ff_nntile, llama_ff_nntile.next_tag
