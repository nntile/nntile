from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        notrans, trans, Tensor_fp32
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
        embed_dim = config["hidden_size"]
        interm_size = config["interm_size"]
        gemm_ndim = 1
        # Initial linear layer that converts input to internal shape
        new_layer, next_tag = Linear.generate_simple_mpiroot(x, "L", notrans,
                gemm_ndim, [interm_size], [interm_size], next_tag)
        print("Layer 0 shape", new_layer.w.value.shape, new_layer.y.value.shape)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        
        new_layer, next_tag = Act.generate_simple(activations[-1], config["activation_function"],
                                                  next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer, next_tag = Linear.generate_simple_mpiroot(activations[-1], "L", notrans,
                gemm_ndim, [embed_dim], [embed_dim], next_tag)
        print("Layer 1 shape", new_layer.w.value.shape, new_layer.y.value.shape)
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
    def from_torch(torch_mlp, x: TensorMoments, batch_size: int, config: Dict, next_tag: int):
        '''
        torch_mlp is PyTorch MLP where no biases in linear layers
        '''
        print("Call from torch static method")
        gpt2mlp_nntile = GPT2MLP(x, config, next_tag)
        torch_params = list(torch_mlp.parameters())
        for i, p in enumerate(gpt2mlp_nntile.parameters):
            p.value.from_array(torch_params[i].detach().cpu().numpy())
        return gpt2mlp_nntile, gpt2mlp_nntile.next_tag