from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        TransOp, trans, notrans, copy_async, gemm_async, randn_async
from nntile.layer.base_layer import BaseLayer
from nntile.layer.mlp_mixer import MlpMixer
import numpy as np
from typing import List


class Mixer(BaseLayer):
    side: str
    trans_x: TransOp
    x: TensorMoments
    y_intermediate: TensorMoments
    y: TensorMoments
    ndim: int
    block_mlp_1: MlpMixer
    block_mlp_2: MlpMixer

    # Construct linear layer with all the provided data
    def __init__(self, trans_x: TransOp, x: TensorMoments, \
            y: TensorMoments, mlp_1: MlpMixer, mlp_2: MlpMixer, ndim: int):
        # Check parameter side
        # if side != 'L' and side != 'R':
        #     raise ValueError("side must be either 'L' or 'R'")

        # Check parameter ndim
        if ndim <= 0:
            raise ValueError("ndim must be positive integer")
        
        # Redirect to BaseClass initialization
        super().__init__([x], [y], [], [])

        # Set up local named parameters
        self.trans_x = trans_x
        self.ndim = ndim
        self.x = x
        self.y = y
        self.block_mlp_1 = mlp_1
        self.block_mlp_2 = mlp_2


    # Simple generator for the mixer layer
    @staticmethod
    def generate_simple_mpiroot(x: TensorMoments, trans_x: TransOp, ndim: int, next_tag: int):

        # Define mlp_mixer layers
        layer1_add_shape = x.value.shape[0] * 4
        mlp_layer1, next_tag = MlpMixer.generate_simple_mpiroot(x, 'R', trans_x, 1, [layer1_add_shape], [layer1_add_shape], next_tag)
        
        layer2_add_shape = x.value.shape[-1] * 4
        mlp_layer2, next_tag = MlpMixer.generate_simple_mpiroot(x, 'L', trans_x, 1, [layer2_add_shape], [layer2_add_shape], next_tag)

        y_shape = x.value.shape
        y_tile = x.value.basetile_shape

        # Define Y
        y_traits = TensorTraits(y_shape, y_tile)

        y_distr = [0] * y_traits.grid.nelems
        y_value = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_value.next_tag
        # Create gradient of Y with the same traits and distribution as Y
        y_grad = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_grad.next_tag

        # Define Y as TensorMoments
        y = TensorMoments(y_value, y_grad, True)

        layer = Mixer(trans_x, x, y, mlp_layer1, mlp_layer2, ndim)

        return layer, next_tag
    

    # Forward propagation of the mixer layer
    def forward_async(self):
        self.block_mlp_1.forward_async()
        copy_async(self.block_mlp_1.y.value, self.block_mlp_2.x.value)
        self.block_mlp_2.forward_async()
        copy_async(self.block_mlp_2.y.value, self.y.value)