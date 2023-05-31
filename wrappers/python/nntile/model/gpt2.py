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
    def __init__(self, input_ids: TensorMoments, positional_ids: TensorMoments, config, next_tag: int):
        # Check parameter side
        vocab_size = config["vocab_size"]
        embed_dim = config["embed_dim"]
        max_position_embeddings = config["max_position_embeddings"]
        activations = [input_ids, positional_ids]
        layers = []
        wte_layer, next_tag = Embedding.generate_simple(input_ids.value, Tensor_fp32, 2, 
                                                        vocab_size, embed_dim, embed_dim,
                                                        embed_dim, next_tag) # config.vocab_size, self.embed_dim
        layers.append(wte_layer)
        activations.append(wte_layer.y)
        
        wpe_layer, next_tag = Embedding.generate_simple(positional_ids.value, Tensor_fp32, 1,
                                                        max_position_embeddings, embed_dim,
                                                        embed_dim, embed_dim, next_tag) # config.max_position_embeddings, self.embed_dim
        layers.append(wpe_layer)
        activations.append(wpe_layer.y)

        
        self.next_tag = next_tag
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    @staticmethod
    def from_torch(torch_gpt2, batch_size: int, seq_len: int, next_tag: int):
        config = {
            "vocab_size": torch_gpt2.wte.weight.shape[0],
            "embed_dim": torch_gpt2.wte.weight.shape[1],
            "max_position_embeddings": torch_gpt2.wpe.weight.shape[0]
        }
        positional_ids_traits = TensorTraits([seq_len], [seq_len])
        positional_ids_distr = [0] * positional_ids_traits.grid.nelems
        positional_ids_value = Tensor_int64(positional_ids_traits, positional_ids_distr, next_tag)
        next_tag = positional_ids_value.next_tag
        positional_ids_value.from_array(np.array(np.arange(seq_len), order="F", dtype=np.int64))
        positional_ids = TensorMoments(positional_ids_value, None, False)
        
        x_traits = TensorTraits([batch_size, seq_len], \
        [batch_size, seq_len])
        x_distr = [0] * x_traits.grid.nelems
        x = Tensor_int64(x_traits, x_distr, next_tag)
        next_tag = x.next_tag
        x_grad = None
        x_grad_required = False
        x_moments = TensorMoments(x, x_grad, x_grad_required)

        gpt2_nntile = GPT2(x_moments, positional_ids, config, next_tag)
        for p_nntile, p_torch in zip(gpt2_nntile.parameters[:2], list(torch_gpt2.parameters())[:2]):
            print(p_torch.shape)
            p_nntile.value.from_array(p_torch.detach().numpy().T)

        return gpt2_nntile, gpt2_nntile.next_tag
