# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt2.py
# GPT2 model of NNTile Python package
#
# @version 1.0.0

from nntile.tensor import (
    TensorTraits,
    Tensor,
    TensorOrNone,
    TensorMoments,
    notrans,
    trans,
    Tensor_fp32,
    Tensor_fp32_fast_tf32,
    Tensor_int64,
    Tensor_bool,
)
from nntile.model.base_model import BaseModel
from nntile.layer import Linear, Embedding, AddSlice, LayerNorm, Attention, Act
from nntile.layer import FlashAttention, AttentionSingleHead
import numpy as np
from typing import List, Dict
from nntile.layer.add import Add
import torch


class GPT2Config(Dict):
    def __init__(
        self,
        vocab_size: int,
        vocab_embed_dim_tile: int,
        embed_dim: int,
        embed_dim_tile: int,
        max_position_embeddings: int,
        inner_dim: int,
        inner_dim_tile: int,
        layer_norm_epsilon: float,
        num_hidden_layers: int,
        n_head: int,
        n_head_tile: int,
        activation_function: str,
        flashattention: bool = True,
        use_redux: bool = False,
        dtype: str = "fp32",
    ):
        self["vocab_size"] = vocab_size
        self["vocab_embed_dim_tile"] = vocab_embed_dim_tile
        self["embed_dim"] = embed_dim
        self["embed_dim_tile"] = embed_dim_tile
        self["max_position_embeddings"] = max_position_embeddings
        self["inner_dim"] = inner_dim
        self["inner_dim_tile"] = inner_dim_tile
        self["layer_norm_epsilon"] = layer_norm_epsilon
        self["num_hidden_layers"] = num_hidden_layers
        self["n_head"] = n_head
        self["n_head_tile"] = n_head_tile
        self["activation_function"] = activation_function
        self["flashattention"] = flashattention
        self["redux"] = use_redux
        self["dtype"] = dtype

    def __getattr__(self, attr):
        return self[attr]


class GPT2MLP(BaseModel):
    next_tag: int

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, config: GPT2Config, next_tag: int):
        # Init activations and list of layers
        activations = [x]
        layers = []
        embed_dim = config["embed_dim"]
        embed_dim_tile = config["embed_dim_tile"]
        inner_dim = config["inner_dim"]
        inner_dim_tile = config["inner_dim_tile"]
        activation_function = config["activation_function"]
        redux = config["redux"]
        gemm_ndim = 1
        # Initial linear layer that converts input to internal shape
        new_layer, next_tag = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [inner_dim],
            [inner_dim_tile],
            next_tag,
            redux=redux,
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer, next_tag = Act.generate_simple(
            activations[-1], activation_function, next_tag
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer, next_tag = Linear.generate_simple(
            activations[-1],
            "R",
            notrans,
            gemm_ndim,
            [embed_dim],
            [embed_dim_tile],
            next_tag,
            redux=redux,
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        self.next_tag = next_tag
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    # Randomly init all linear layers
    def init_randn_async(self):
        for l in self.layers:
            if type(l) is Linear:
                l.init_randn_async()

    @staticmethod
    def from_torch(
        torch_mlp, x: TensorMoments, config: GPT2Config, next_tag: int
    ):
        """
        torch_mlp is PyTorch MLP where no biases in linear layers
        """
        gpt2mlp_nntile = GPT2MLP(x, config, next_tag)
        torch_params = list(torch_mlp.parameters())
        for i, p in enumerate(gpt2mlp_nntile.parameters):
            p.value.from_array(torch_params[i].cpu().detach().numpy().T)
        return gpt2mlp_nntile, gpt2mlp_nntile.next_tag


class GPT2Model(BaseModel):
    next_tag: int

    # Construct model with all the provided data
    def __init__(
        self,
        input_ids: TensorMoments,
        positional_ids: TensorMoments,
        config: GPT2Config,
        next_tag: int,
    ):
        # Check parameter side
        vocab_size = config["vocab_size"]
        vocab_embed_dim_tile = config["vocab_embed_dim_tile"]
        self.embed_dim = config["embed_dim"]
        embed_dim_tile = config["embed_dim_tile"]
        max_position_embeddings = config["max_position_embeddings"]
        inner_dim = config["inner_dim"]
        inner_dim_tile = config["inner_dim_tile"]
        layer_norm_epsilon = config["layer_norm_epsilon"]
        num_hidden_layers = config["num_hidden_layers"]
        self.n_head = config["n_head"]
        n_head_tile = config["n_head_tile"]
        flashattention = config["flashattention"]
        redux = config["redux"]
        self.dtype = config["dtype"]

        if self.dtype not in ["fp32", "tf32"]:
            raise TypeError("Only fp32 and tf32 are supported for weight type")

        if self.n_head == 1:
            print("Set 1 head")
            AttLayer = AttentionSingleHead
        elif flashattention:
            AttLayer = FlashAttention
        else:
            AttLayer = Attention
        seq_len = input_ids.value.shape[0]
        seq_len_tile = input_ids.value.basetile_shape[0]
        activations = [input_ids, positional_ids]
        layers = []
        mask_traits = TensorTraits(
            (seq_len, seq_len), (seq_len_tile, seq_len_tile)
        )
        mask_distr = [0] * mask_traits.grid.nelems
        self.mask = Tensor_bool(mask_traits, mask_distr, next_tag)
        next_tag = self.mask.next_tag
        mask_np = np.array(
            np.triu(np.ones((seq_len, seq_len))), dtype=bool, order="F"
        )
        self.mask.from_array(mask_np)

        if self.dtype == "fp32":
            wte_layer, next_tag = Embedding.generate_simple(
                input_ids.value,
                Tensor_fp32,
                0,
                vocab_size,
                self.embed_dim,
                embed_dim_tile,
                vocab_embed_dim_tile,
                next_tag,
            )
        elif self.dtype == "tf32":
            wte_layer, next_tag = Embedding.generate_simple(
                input_ids.value,
                Tensor_fp32_fast_tf32,
                0,
                vocab_size,
                self.embed_dim,
                embed_dim_tile,
                vocab_embed_dim_tile,
                next_tag,
            )

        layers.append(wte_layer)
        activations.extend(wte_layer.activations_output)

        if self.dtype == "fp32":
            wpe_layer, next_tag = Embedding.generate_simple(
                positional_ids.value,
                Tensor_fp32,
                0,
                max_position_embeddings,
                self.embed_dim,
                embed_dim_tile,
                vocab_embed_dim_tile,
                next_tag,
            )
        elif self.dtype == "tf32":
            wpe_layer, next_tag = Embedding.generate_simple(
                positional_ids.value,
                Tensor_fp32_fast_tf32,
                0,
                max_position_embeddings,
                self.embed_dim,
                embed_dim_tile,
                vocab_embed_dim_tile,
                next_tag,
            )

        layers.append(wpe_layer)
        activations.extend(wpe_layer.activations_output)

        add_slice_layer, next_tag = AddSlice.generate_simple(
            activations[-2], activations[-1], 2, next_tag, redux=redux
        )
        layers.append(add_slice_layer)
        activations.extend(add_slice_layer.activations_output)

        for h_idx in range(num_hidden_layers):
            l_norm, next_tag = LayerNorm.generate_simple(
                activations[-1], 0, layer_norm_epsilon, next_tag, redux=redux
            )
            layers.append(l_norm)
            activations.extend(l_norm.activations_output)

            if self.n_head == 1:
                attn_layer, next_tag = AttLayer.generate_simple(
                    activations[-1],
                    activations[-1],
                    activations[-1],
                    next_tag,
                    True,
                    self.mask,
                    redux=redux,
                )
            else:
                attn_layer, next_tag = AttLayer.generate_simple(
                    activations[-1],
                    activations[-1],
                    activations[-1],
                    self.n_head,
                    n_head_tile,
                    next_tag,
                    True,
                    self.mask,
                    redux=redux,
                )
            layers.append(attn_layer)
            activations.extend(attn_layer.activations_output)

            new_layer, next_tag = Add.generate_simple(
                activations[-3], activations[-1], next_tag
            )
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)

            l_norm, next_tag = LayerNorm.generate_simple(
                activations[-1], 0, layer_norm_epsilon, next_tag, redux=redux
            )
            layers.append(l_norm)
            activations.extend(l_norm.activations_output)

            gpt_block = GPT2MLP(activations[-1], config, next_tag)
            next_tag = gpt_block.next_tag

            activations.extend(gpt_block.activations[1:])
            layers.extend(gpt_block.layers)

            new_layer, next_tag = Add.generate_simple(
                activations[-5], activations[-1], next_tag
            )
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)

        l_norm, next_tag = LayerNorm.generate_simple(
            activations[-1], 0, layer_norm_epsilon, next_tag, redux=redux
        )

        layers.append(l_norm)
        activations.extend(l_norm.activations_output)

        lm_head_layer, next_tag = Linear.generate_simple(
            activations[-1],
            "R",
            notrans,
            1,
            [vocab_size],
            [vocab_size],
            next_tag,
            False,
            redux=redux,
        )

        layers.append(lm_head_layer)
        activations.extend(lm_head_layer.activations_output)

        self.next_tag = next_tag
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    def to_torch(self, base_torch_model):
        nntile_p_idx = 0
        attn_embed_dim = self.embed_dim
        attn_nheads = self.n_head
        attn_head_size = attn_embed_dim // attn_nheads
        for name, p in base_torch_model.named_parameters():
            layer_name = name.split(".")[-2]
            if layer_name in ("lm_head",):
                p_np = np.array(np.zeros(p.shape, dtype=np.float32), order="F")
                self.parameters[nntile_p_idx].value.to_array(p_np)
                p.data = torch.from_numpy(p_np)
                nntile_p_idx += 1
            elif layer_name == "c_attn" and name.split(".")[-1] == "weight":
                # p_torch_np = p_torch.cpu().detach().numpy()
                # Read Q, K and V weights
                for i_tensor in range(3):
                    p_nntile_np = np.array(
                        np.zeros(
                            self.parameters[nntile_p_idx].value.shape,
                            dtype=np.float32,
                        ),
                        order="F",
                    )
                    self.parameters[nntile_p_idx].value.to_array(p_nntile_np)
                    init_shape = p[
                        :,
                        i_tensor * attn_embed_dim : (i_tensor + 1)
                        * attn_embed_dim,
                    ].T.shape
                    cur_tensor = torch.from_numpy(p_nntile_np).reshape(
                        init_shape
                    )

                    p.data[
                        :,
                        i_tensor * attn_embed_dim : (i_tensor + 1)
                        * attn_embed_dim,
                    ] = cur_tensor.T
                    nntile_p_idx += 1
            elif layer_name == "c_attn" and name.split(".")[-1] == "bias":
                # p_torch_np = p_torch.cpu().detach().numpy()
                # Read Q, K and V biases
                for i_tensor in range(3):
                    p_nntile_np = np.array(
                        np.zeros(
                            self.parameters[nntile_p_idx].value.shape,
                            dtype=np.float32,
                        ),
                        order="F",
                    )
                    self.parameters[nntile_p_idx].value.to_array(p_nntile_np)
                    cur_tensor = torch.from_numpy(p_nntile_np)
                    p.data[
                        i_tensor * attn_embed_dim : (i_tensor + 1)
                        * attn_embed_dim
                    ] = cur_tensor.T.reshape(-1)
                    nntile_p_idx += 1
            elif layer_name == "c_proj" and name.split(".")[-3] == "attn":
                # p_torch_np = p_torch.cpu().detach().numpy()
                p_nntile = self.parameters[nntile_p_idx].value
                p_nntile_np = np.array(
                    np.zeros(p_nntile.shape, dtype=np.float32), order="F"
                )
                p_nntile.to_array(p_nntile_np)
                if name.split(".")[-1] == "weight":
                    init_shape = p.T.shape
                    cur_tensor = torch.from_numpy(p_nntile_np)
                    p.data = cur_tensor.reshape(init_shape).T

                    #     cur_tensor = p.T.reshape(attn_embed_dim, attn_nheads, \
                    #             attn_head_size)
                    #     cur_tensor.data = torch.from_numpy(p_nntile_np)
                    #     p_nntile.value.from_array(p_torch_np.T \
                    #             .reshape(attn_embed_dim, attn_nheads, \
                    #             attn_head_size))
                    nntile_p_idx += 1
                elif name.split(".")[-1] == "bias":
                    #     p_nntile = gpt2_nntile.parameters[nntile_p_idx]
                    #     p_nntile.value.from_array(p_torch_np)
                    p.data = torch.from_numpy(p_nntile_np)
                    nntile_p_idx += 1
            else:
                p_np = np.array(
                    np.zeros(
                        self.parameters[nntile_p_idx].value.shape,
                        dtype=np.float32,
                    ),
                    order="F",
                )
                self.parameters[nntile_p_idx].value.to_array(p_np)
                p.data = torch.from_numpy(p_np.T)
                nntile_p_idx += 1

    @staticmethod
    def from_torch(
        torch_gpt2,
        batch_size: int,
        batch_size_tile: int,
        seq_len: int,
        seq_len_tile: int,
        config: GPT2Config,
        next_tag: int,
    ):
        positional_ids_traits = TensorTraits([seq_len], [seq_len_tile])
        positional_ids_distr = [0] * positional_ids_traits.grid.nelems
        positional_ids_value = Tensor_int64(
            positional_ids_traits, positional_ids_distr, next_tag
        )
        next_tag = positional_ids_value.next_tag
        positional_ids_value.from_array(
            np.array(np.arange(seq_len), order="F", dtype=np.int64)
        )
        positional_ids = TensorMoments(positional_ids_value, None, False)

        x_traits = TensorTraits(
            [seq_len, batch_size], [seq_len_tile, batch_size_tile]
        )
        x_distr = [0] * x_traits.grid.nelems
        x = Tensor_int64(x_traits, x_distr, next_tag)
        next_tag = x.next_tag
        x_grad = None
        x_grad_required = False
        x_moments = TensorMoments(x, x_grad, x_grad_required)

        gpt2_nntile = GPT2Model(x_moments, positional_ids, config, next_tag)
        nntile_p_idx = 0
        attn_embed_dim = config["embed_dim"]
        attn_nheads = config["n_head"]
        attn_head_size = attn_embed_dim // attn_nheads
        for name, p_torch in torch_gpt2.named_parameters():
            layer_name = name.split(".")[-2]
            if layer_name in ("lm_head",):
                p_nntile = gpt2_nntile.parameters[nntile_p_idx]
                p_nntile.value.from_array(p_torch.cpu().detach().numpy())
                nntile_p_idx += 1
            elif layer_name == "c_attn" and name.split(".")[-1] == "weight":
                p_torch_np = p_torch.cpu().detach().numpy()
                # Read Q, K and V weights
                for i_tensor in range(3):
                    p_nntile = gpt2_nntile.parameters[nntile_p_idx]
                    if attn_nheads == 1:
                        p_nntile.value.from_array(
                            p_torch_np[
                                :,
                                i_tensor * attn_embed_dim : (i_tensor + 1)
                                * attn_embed_dim,
                            ].T
                        )
                    else:
                        p_nntile.value.from_array(
                            p_torch_np[
                                :,
                                i_tensor * attn_embed_dim : (i_tensor + 1)
                                * attn_embed_dim,
                            ].T.reshape(
                                attn_nheads, attn_head_size, attn_embed_dim
                            )
                        )
                    nntile_p_idx += 1

            elif layer_name == "c_attn" and name.split(".")[-1] == "bias":
                p_torch_np = p_torch.cpu().detach().numpy()
                # Read Q, K and V biases
                for i_tensor in range(3):
                    p_nntile = gpt2_nntile.parameters[nntile_p_idx]
                    if attn_nheads == 1:
                        p_nntile.value.from_array(
                            p_torch_np[
                                i_tensor * attn_embed_dim : (i_tensor + 1)
                                * attn_embed_dim
                            ]
                        )
                    else:
                        p_nntile.value.from_array(
                            p_torch_np[
                                i_tensor * attn_embed_dim : (i_tensor + 1)
                                * attn_embed_dim
                            ]
                            .reshape(attn_nheads, attn_head_size)
                            .T
                        )
                    nntile_p_idx += 1
            elif layer_name == "c_proj" and name.split(".")[-3] == "attn":
                p_torch_np = p_torch.cpu().detach().numpy()
                if name.split(".")[-1] == "weight":
                    p_nntile = gpt2_nntile.parameters[nntile_p_idx]
                    if attn_nheads == 1:
                        p_nntile.value.from_array(p_torch_np.T)
                    else:
                        p_nntile.value.from_array(
                            p_torch_np.T.reshape(
                                attn_embed_dim, attn_nheads, attn_head_size
                            )
                        )
                    nntile_p_idx += 1
                elif name.split(".")[-1] == "bias":
                    p_nntile = gpt2_nntile.parameters[nntile_p_idx]
                    p_nntile.value.from_array(p_torch_np)
                    nntile_p_idx += 1
            else:
                p_nntile = gpt2_nntile.parameters[nntile_p_idx]
                p_nntile.value.from_array(p_torch.cpu().detach().numpy().T)
                nntile_p_idx += 1

        return gpt2_nntile, gpt2_nntile.next_tag

    def set_input(self, x: Tensor):
        expected_shape = self.activations[0].value.shape
        if x.shape != expected_shape:
            raise Exception(
                "Mismatch shapes. Got: ", x.shape, " Expected: ", expected_shape
            )

        self.activations[0].value = x

    def get_output(self) -> Tensor:
        return self.activations[-1].value

    def set_output_grad(self, grad):
        expected_shape = self.activations[-1].value.shape
        if grad.shape != expected_shape:
            raise Exception(
                "Mismatch shapes. Got: ",
                grad.shape,
                " Expected: ",
                expected_shape,
            )

        self.activations[-1].grad = grad

    def get_input_grad(self):
        return self.activations[0].grad

    def forward(self, x: Tensor) -> Tensor:
        self.set_input(x)
        self.forward_async()
        return self.get_output()

    def unregister(self):
        super().unregister()
        if self.mask:
            self.mask.unregister()
