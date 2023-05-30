# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt2.py
# GPT2 model of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-05-29

from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        notrans, trans, Tensor_fp32, Tensor_int64
from nntile.model.base_model import BaseModel
from nntile.layer.linear import Linear
from nntile.layer.embedding import Embedding
import numpy as np
from typing import List

class GPT2(BaseModel):
    next_tag: int

    # Construct model with all the provided data
    def __init__(self, input_ids: Tensor_int64, positional_ids: Tensor_int64, config, next_tag: int):
        # Check parameter side
        vocab_size = config["vocab_size"]
        embed_dim = config["embed_dim"]
        max_position_embeddings = config["max_position_embeddings"]
        activations = [input_ids, positional_ids]
        self.positional_ids = positional_ids
        layers = []
        wte_layer, next_tag = Embedding.generate_simple(input_ids, Tensor_fp32, 1, 
                                                        vocab_size, embed_dim, embed_dim,
                                                        embed_dim, next_tag) # config.vocab_size, self.embed_dim
        layers.append(wte_layer)
        activations.append(wte_layer.y.value)
        
        wpe_layer, next_tag = Embedding.generate_simple(positional_ids, Tensor_fp32, 1,
                                                        max_position_embeddings, embed_dim,
                                                        embed_dim, embed_dim, next_tag) # config.max_position_embeddings, self.embed_dim
        layers.append(wpe_layer)
        activations.append(wpe_layer.y.value)

        
        self.next_tag = next_tag
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    # Randomly init all linear layers
    # def init_randn_async(self):
    #     for l in self.layers:
    #         if type(l) is Linear:
    #             l.init_randn_async()

    @staticmethod
    def from_torch(torch_gpt2, input_ids, next_tag: int):
        config = {
            "vocab_size": torch_gpt2.wte.weight.shape[0],
            "embed_dim": torch_gpt2.wte.weight.shape[1],
            "max_position_embeddings": torch_gpt2.wpe.weight.shape[0]
        }
        positional_ids_traits = TensorTraits([input_ids.shape[-1]], [input_ids.shape[-1]])
        positional_ids_distr = [0] * positional_ids_traits.grid.nelems
        positional_ids = Tensor_int64(positional_ids_traits, positional_ids_distr, next_tag)
        next_tag = positional_ids.next_tag
        positional_ids.from_array(np.array(np.arange(input_ids.shape[-1]), order="F", dtype=np.int64))
        gpt2_nntile = GPT2(input_ids, positional_ids, config, next_tag)
        for p_nntile, p_torch in zip(gpt2_nntile.parameters[:2], list(torch_gpt2.parameters())[:2]):
            print(p_torch.shape)
            p_nntile.value.from_array(p_torch.detach().numpy().T)

        return gpt2_nntile, gpt2_nntile.next_tag
    
    def unregister(self):
        self.positional_ids.unregister()
        for l in self.layers:
            l.unregister()
