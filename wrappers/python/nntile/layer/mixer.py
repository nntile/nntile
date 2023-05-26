from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        TransOp, trans, notrans, add_async, copy_async, gemm_async, randn_async
from nntile.layer.base_layer import BaseLayer
from nntile.layer.layer_norm import LayerNorm
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
    norm_1: LayerNorm
    norm_2: LayerNorm
    block_mlp_1: MlpMixer
    block_mlp_2: MlpMixer

    # Construct mixer layer with all the provided data
    def __init__(self, trans_x: TransOp, x: TensorMoments, \
            y: TensorMoments, mlp_1: MlpMixer, mlp_2: MlpMixer, ndim: int):

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
        mlp_layer1_add_shape = x.value.shape[0] * 4
        mlp_layer1, next_tag = MlpMixer.generate_simple_mpiroot(x, 'R', trans_x, 1, [mlp_layer1_add_shape], [mlp_layer1_add_shape], next_tag)
        
        y_shape = x.value.shape
        y_tile = x.value.basetile_shape


        # Define Y_tmp
        y_tmp_traits = TensorTraits(y_shape, y_tile)

        y_tmp_distr = [0] * y_tmp_traits.grid.nelems
        y_tmp_value = type(x.value)(y_tmp_traits, y_tmp_distr, next_tag)
        next_tag = y_tmp_value.next_tag
        # Create gradient of Y with the same traits and distribution as Y
        y_tmp_grad = type(x.value)(y_tmp_traits, y_tmp_distr, next_tag)
        next_tag = y_tmp_grad.next_tag

        # Define Y as TensorMoments
        y_tmp = TensorMoments(y_tmp_value, y_tmp_grad, True)

        mlp_layer2_add_shape = x.value.shape[-1] * 4
        mlp_layer2, next_tag = MlpMixer.generate_simple_mpiroot(y_tmp, 'L', trans_x, 1, [mlp_layer2_add_shape], [mlp_layer2_add_shape], next_tag)


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
        add_async(1.0, self.block_mlp_1.x.value, 1.0, self.block_mlp_1.y.value)
        copy_async(self.block_mlp_1.y.value, self.block_mlp_2.x.value)
        self.block_mlp_2.forward_async()
        add_async(1.0, self.block_mlp_2.x.value, 1.0, self.block_mlp_2.y.value)
        copy_async(self.block_mlp_2.y.value, self.y.value)