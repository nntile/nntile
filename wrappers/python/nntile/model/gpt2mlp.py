# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt2mlp.py
# GPT2MLP block for GPT2 model of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-06-15

from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        notrans, trans, Tensor_fp32, clear_async
from nntile.model.base_model import BaseModel
from nntile.layer.linear import Linear
from nntile.layer.act import Act
import numpy as np
from typing import List, Dict

class GPT2MLP(BaseModel):
    next_tag: int

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, config: Dict, next_tag: int):
        # Init activations and list of layers
        activations = [x]
        layers = []
        embed_dim = config["embed_dim"]
        embed_dim_tile = config["embed_dim_tile"]
        interm_size = config["interm_size"]
        interm_size_tile = config["interm_size_tile"]
        gemm_ndim = 1
        # Initial linear layer that converts input to internal shape
        new_layer, next_tag = Linear.generate_simple_mpiroot(x, "R", notrans,
                gemm_ndim, [interm_size], [interm_size_tile], next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        
        new_layer, next_tag = Act.generate_simple(activations[-1], \
                config["activation_function"], next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer, next_tag = Linear.generate_simple_mpiroot(activations[-1], \
                "R", notrans, gemm_ndim, [embed_dim], [embed_dim_tile], \
                next_tag)
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
    def from_torch(torch_mlp, x: TensorMoments, config: Dict, next_tag: int):
        '''
        torch_mlp is PyTorch MLP where no biases in linear layers
        '''
        gpt2mlp_nntile = GPT2MLP(x, config, next_tag)
        torch_params = list(torch_mlp.parameters())
        for i, p in enumerate(gpt2mlp_nntile.parameters):
            p.value.from_array(torch_params[i].cpu().detach().numpy().T)
        return gpt2mlp_nntile, gpt2mlp_nntile.next_tag

