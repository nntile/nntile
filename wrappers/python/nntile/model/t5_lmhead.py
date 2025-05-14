# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/t5_lmhead.py
# T5 LMHead submodule of NNTile Python package
#
# @version 1.1.0

from nntile.model.base_model import BaseModel
from nntile.tensor import TensorMoments, notrans
import nntile.utils.constructors as nntc
from nntile.layer.linear import Linear
from nntile.layer.act import Act
from nntile.model.t5_config import T5ConfigNNTile
import torch
from transformers.models.t5.modeling_t5 import T5Config as T5ConfigTorch
from transformers.models.t5.modeling_t5 import T5ClassificationHead as T5ClassificationHeadTorch
from nntile.tensor import to_numpy
        

class T5ClassificationHead(BaseModel):
    next_tag: int

    def __init__(self, x: TensorMoments, config: T5ConfigNNTile, num_labels: int, next_tag: int):
        """Head for sentence-level classification tasks."""
        activations = [x]
        layers = []
        self.d_model = config.d_model
        self.d_model_tile = config.d_model_tile
        self.num_labels = num_labels
        self.redux = config.redux

        gemm_ndim = 1

        # First linear layer (dense)
        dense, next_tag = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [self.d_model],
            [self.d_model_tile],
            next_tag,
            redux=self.redux,
            bias=True,
        )

        layers.append(dense)
        activations.extend(dense.activations_output)

        # Using gelutanh activation since tanh is not directly implemented
        # This provides a non-linear activation similar to tanh
        act_fn_layer, next_tag = Act.generate_simple(
            activations[-1], "gelutanh", next_tag
        )

        layers.append(act_fn_layer)
        activations.extend(act_fn_layer.activations_output)

        # Output projection layer
        out_proj, next_tag = Linear.generate_simple(
            activations[-1],
            "R",
            notrans,
            gemm_ndim,
            [self.num_labels],
            [self.num_labels],  # Using num_labels as tile size
            next_tag,
            redux=self.redux,
            bias=True,
        )

        layers.append(out_proj)
        activations.extend(out_proj.activations_output)

        # Store layers for direct access
        self.dense = dense
        self.act_fn_layer = act_fn_layer
        self.out_proj = out_proj
        
        self.next_tag = next_tag

        # Initialize base model
        super().__init__(activations, layers, config)

    @classmethod
    def from_torch(cls, torch_head, x: TensorMoments, config: T5ConfigNNTile, num_labels: int, next_tag: int):
        """Create T5ClassificationHead from PyTorch model"""
        t5_head_nntile = cls(x, config, num_labels, next_tag)
        
        # Transfer parameters from PyTorch model to NNTile
        # First dense layer
        t5_head_nntile.dense.w.value.from_array(torch_head.dense.weight.cpu().detach().numpy())
        t5_head_nntile.dense.b.value.from_array(torch_head.dense.bias.cpu().detach().numpy())
        
        # Output projection layer
        t5_head_nntile.out_proj.w.value.from_array(torch_head.out_proj.weight.cpu().detach().numpy())
        t5_head_nntile.out_proj.b.value.from_array(torch_head.out_proj.bias.cpu().detach().numpy())
        
        return t5_head_nntile, t5_head_nntile.next_tag
    def to_torch(self):
        """Convert NNTile T5ClassificationHead to PyTorch T5ClassificationHead"""
        # Import inside function to avoid circular imports
        
        # Create PyTorch config
        torch_config = T5ConfigTorch(
            d_model=self.config.d_model,
            num_labels=self.num_labels
        )
        
        # Create PyTorch T5ClassificationHead
        torch_head = T5ClassificationHeadTorch(torch_config)
        
        # Copy parameters from NNTile to PyTorch
        torch_head.dense.weight.data = torch.tensor(to_numpy(self.dense.w.value), requires_grad=True)
        torch_head.dense.bias.data = torch.tensor(to_numpy(self.dense.b.value), requires_grad=True)
        torch_head.out_proj.weight.data = torch.tensor(to_numpy(self.out_proj.w.value), requires_grad=True)
        torch_head.out_proj.bias.data = torch.tensor(to_numpy(self.out_proj.b.value), requires_grad=True)
        
        return torch_head
    
    def to_torch_with_grads(self):
        """Convert NNTile T5ClassificationHead to PyTorch T5ClassificationHead with gradients"""
        torch_head = self.to_torch()
        
        # Copy gradients from NNTile to PyTorch
        torch_head.dense.weight.grad = torch.tensor(nntc.to_numpy(self.dense.w.grad))
        torch_head.dense.bias.grad = torch.tensor(nntc.to_numpy(self.dense.b.grad))
        torch_head.out_proj.weight.grad = torch.tensor(nntc.to_numpy(self.out_proj.w.grad))
        torch_head.out_proj.bias.grad = torch.tensor(nntc.to_numpy(self.out_proj.b.grad))
        
        return torch_head
