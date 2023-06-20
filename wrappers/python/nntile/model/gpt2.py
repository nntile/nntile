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
from nntile.layer.add_slice import AddSlice
from nntile.layer.layer_norm import LayerNorm
from nntile.layer.attention import Attention
import numpy as np
from typing import List
from nntile.model.gpt2mlp import GPT2MLP
from nntile.layer.add import Add

class GPT2(BaseModel):
    next_tag: int

    # Construct model with all the provided data
    def __init__(self, input_ids: TensorMoments, positional_ids: TensorMoments, config, next_tag: int):
        # Check parameter side
        vocab_size = config["vocab_size"]
        embed_dim = config["embed_dim"]
        max_position_embeddings = config["max_position_embeddings"]
        layer_norm_epsilon = config["layer_norm_epsilon"]
        hidden_size = config["hidden_size"]
        activations = [input_ids, positional_ids]
        layers = []
        wte_layer, next_tag = Embedding.generate_simple(input_ids.value, Tensor_fp32, 0, 
                                                        vocab_size, embed_dim, embed_dim,
                                                        embed_dim, next_tag) # config.vocab_size, self.embed_dim
        layers.append(wte_layer)
        activations.extend(wte_layer.activations_output)
        
        wpe_layer, next_tag = Embedding.generate_simple(positional_ids.value, Tensor_fp32, 0,
                                                        max_position_embeddings, embed_dim,
                                                        embed_dim, embed_dim, next_tag) # config.max_position_embeddings, self.embed_dim
        layers.append(wpe_layer)
        activations.extend(wpe_layer.activations_output)

        add_slice_layer, next_tag = AddSlice.generate_simple(activations[-2], activations[-1], 2, next_tag)
        
        layers.append(add_slice_layer)
        activations.extend(add_slice_layer.activations_output)

        for h_idx in range(config["num_hidden_layers"]):
            l_norm, next_tag = LayerNorm.generate_simple(activations[-1], 0, layer_norm_epsilon, next_tag)
            layers.append(l_norm)
            activations.extend(l_norm.activations_output)

            # attn_layer, next_tag = Attention.generate_simple_mpiroot(activations[-1],
            #                                                          activations[-1],
            #                                                          activations[-1],
            #                                                          config["n_head"], next_tag)
            # layers.append(attn_layer)
            # activations.extend(attn_layer.activations_output)

            new_layer, next_tag = Add.generate_simple(activations[-2], activations[-1],
                                                  next_tag)
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)

            l_norm, next_tag = LayerNorm.generate_simple(activations[-1], 0, layer_norm_epsilon, next_tag)
            layers.append(l_norm)
            activations.extend(l_norm.activations_output)

            inner_dim = config["n_inner"] if config["n_inner"] is not None else 4 * hidden_size
            config["interm_size"] = inner_dim
            gpt_block = GPT2MLP(activations[-1], config, next_tag)
            next_tag = gpt_block.next_tag

            activations.extend(gpt_block.activations[1:])
            layers.extend(gpt_block.layers)
            

            new_layer, next_tag = Add.generate_simple(activations[-5], activations[-1],
                                                  next_tag)
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)



        l_norm, next_tag = LayerNorm.generate_simple(activations[-1], 0, layer_norm_epsilon, next_tag)

        layers.append(l_norm)
        activations.extend(l_norm.activations_output)

        lm_head_layer, next_tag = Linear.generate_simple_mpiroot(activations[-1], "R", notrans, 1, [vocab_size], [vocab_size], next_tag)

        layers.append(lm_head_layer)
        activations.extend(lm_head_layer.activations_output)

        self.next_tag = next_tag
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    @staticmethod
    def from_torch(torch_gpt2, batch_size: int, seq_len: int, config, next_tag: int):
        # config = {
        #     "vocab_size": torch_gpt2.transformer.wte.weight.shape[0],
        #     "embed_dim": torch_gpt2.transformer.wte.weight.shape[1],
        #     "max_position_embeddings": torch_gpt2.transformer.wpe.weight.shape[0],
        #     "layer_norm_epsilon": layer_norm_eps
        # }
        config = {
            "vocab_size": config.vocab_size,
            "embed_dim": config.n_embd,
            "max_position_embeddings": config.max_position_embeddings,
            "layer_norm_epsilon": config.layer_norm_epsilon,
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "n_inner": config.n_inner,
            "activation_function": "gelutanh",
            "n_head": config.n_head,
        }
        positional_ids_traits = TensorTraits([seq_len], [seq_len])
        positional_ids_distr = [0] * positional_ids_traits.grid.nelems
        positional_ids_value = Tensor_int64(positional_ids_traits, positional_ids_distr, next_tag)
        next_tag = positional_ids_value.next_tag
        positional_ids_value.from_array(np.array(np.arange(seq_len), order="F", dtype=np.int64))
        positional_ids = TensorMoments(positional_ids_value, None, False)
        
        x_traits = TensorTraits([seq_len, batch_size], \
        [seq_len, batch_size])
        x_distr = [0] * x_traits.grid.nelems
        x = Tensor_int64(x_traits, x_distr, next_tag)
        next_tag = x.next_tag
        x_grad = None
        x_grad_required = False
        x_moments = TensorMoments(x, x_grad, x_grad_required)

        gpt2_nntile = GPT2(x_moments, positional_ids, config, next_tag)
        for p_nntile, (name, p_torch) in zip(gpt2_nntile.parameters, list(torch_gpt2.named_parameters())):
            print(p_nntile.value.shape, p_torch.shape, name)
            layer_name = name.split(".")[-2]
            if layer_name in ("lm_head",):
                p_nntile.value.from_array(p_torch.detach().numpy())
            else:
                p_nntile.value.from_array(p_torch.detach().numpy().T)
        # gpt2_nntile.parameters[-1].value.from_array(torch_gpt2.lm_head.weight.data.detach().numpy())

        return gpt2_nntile, gpt2_nntile.next_tag
