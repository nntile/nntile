from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        TransOp, trans, notrans, add_async, copy_async, gelu_async, gemm_async
from nntile.layer.base_layer import BaseLayer
from nntile.layer.layer_norm import LayerNorm
from nntile.layer.linear import Linear
from nntile.layer.act import Act
import numpy as np
from typing import List


class MixerMlp(BaseLayer):
    side: str
    x: TensorMoments
    y: TensorMoments
    linear_1: Linear
    linear_2: Linear
    act: Act


    # Construct MixerMlp layer with all the provided data
    def __init__(self, side: str, x: TensorMoments, y: TensorMoments, 
                 linear_1: Linear, linear_2: Linear, act: Act):
        # Check parameter side
        if side != 'L' and side != 'R':
            raise ValueError("side must be either 'L' or 'R'")
        # Redirect to BaseClass initialization
        super().__init__([x], [y], [], [])
        
        # Set up local named parameters
        self.side = side
        self.x = x
        self.y = y
        self.linear_1 = linear_1
        self.linear_2 = linear_2
        self.act = act


    @staticmethod
    def generate_simple_mpiroot(x: TensorMoments, side: str, next_tag: int):
        if side == 'R':
            add_shape = x.value.shape[0] * 4
            add_basetile_shape = x.value.shape[0] * 4
            init_shape = x.value.shape[0]
        if side == 'L':
            add_shape = x.value.shape[-1] * 4
            add_basetile_shape = x.value.shape[-1] * 4
            init_shape = x.value.shape[-1]
        linear_1_layer, next_tag = Linear.generate_simple_mpiroot(x, side, notrans, 1, [add_shape], [add_basetile_shape], next_tag)
        act_layer, next_tag = Act.generate_simple(linear_1_layer.y, 'gelu', next_tag)
        linear_2_layer, next_tag = Linear.generate_simple_mpiroot(act_layer.y, side, notrans, 1, [init_shape], [init_shape], next_tag)
        layer = MixerMlp(side, x, linear_2_layer.y, linear_1_layer, linear_2_layer, act_layer)

        # Return layer and next tag to be used
        return (layer, next_tag)
    

    def forward_async(self):
        self.linear_1.forward_async()
        self.act.forward_async()
        self.linear_2.forward_async()


class Mixer(BaseLayer):
    x: TensorMoments
    y: TensorMoments
    norm_1: LayerNorm
    norm_2: LayerNorm
    mlp_1: MixerMlp
    mlp_2: MixerMlp

    # Construct mixer layer with all the provided data
    def __init__(self, x: TensorMoments, y: TensorMoments, 
                 norm_1: LayerNorm, norm_2: LayerNorm,
                 mlp_1: MixerMlp, mlp_2: MixerMlp):
        
        # Redirect to BaseClass initialization
        super().__init__([x], [y], [], [])

        # Set up local named parameters
        self.x = x
        self.y = y
        self.norm_1 = norm_1
        self.norm_2 = norm_2
        self.mlp_1 = mlp_1
        self.mlp_2 = mlp_2


    # Simple generator for the mixer layer
    @staticmethod
    def generate_simple_mpiroot(x: TensorMoments, next_tag: int):
        eps = 1e-5
        norm_1_layer, next_tag = LayerNorm.generate_simple(x, 2, eps, next_tag)

        mlp_layer1, next_tag = MixerMlp.generate_simple_mpiroot(norm_1_layer.y, 'R', next_tag)
        
        norm_2_layer, next_tag = LayerNorm.generate_simple(mlp_layer1.y, 2, eps, next_tag)

        mlp_layer2, next_tag = MixerMlp.generate_simple_mpiroot(norm_2_layer.y, 'L', next_tag)

        layer = Mixer(x, mlp_layer2.y, norm_1_layer, norm_2_layer, mlp_layer1, mlp_layer2)

        return layer, next_tag
    

    # Forward propagation of the mixer layer
    def forward_async(self):
        self.norm_1.forward_async()
        self.mlp_1.forward_async()
        add_async(1.0, self.x.value, 1.0, self.mlp_1.y.value)
        self.norm_2.forward_async()
        self.mlp_2.forward_async()
        add_async(1.0, self.norm_2.x.value, 1.0, self.mlp_2.y.value)
