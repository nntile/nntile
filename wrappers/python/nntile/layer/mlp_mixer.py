from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        TransOp, trans, notrans, copy_async, gemm_async, randn_async, gelu_async, \
        gelu_backward_async
from nntile.layer.base_layer import BaseLayer
import numpy as np
from typing import List


class MlpMixer(BaseLayer):
    side: str
    trans_x: TransOp
    x: TensorMoments
    interm_1: TensorMoments
    interm_2: TensorMoments
    y: TensorMoments
    w1: TensorMoments
    w2: TensorMoments
    ndim: int

    # Construct mlp_mixer layer with all the provided data
    def __init__(self, side: str, trans_x: TransOp, x: TensorMoments, \
            y: TensorMoments, w1: TensorMoments, w2: TensorMoments, 
            interm_1: TensorMoments, interm_2: TensorMoments, ndim: int):
        # Check parameter side
        if side != 'L' and side != 'R':
            raise ValueError("side must be either 'L' or 'R'")
        
        # Check parameter ndim
        if ndim <= 0:
            raise ValueError("ndim must be positive integer")
        
        # Redirect to BaseClass initialization
        super().__init__([x], [y], [w1, w2], [interm_1, interm_2])
        
        # Set up local named parameters
        self.side = side
        self.trans_x = trans_x
        self.ndim = ndim
        self.x = x
        self.y = y
        self.interm_1 = interm_1
        self.interm_2 = interm_2
        self.w1 = w1
        self.w2 = w2


    # Simple generator for the mlp_mixer layer
    @staticmethod
    def generate_simple_mpiroot(x: TensorMoments, side: str, trans_x: TransOp,
            ndim: int, add_shape: List[int], add_basetile_shape: List[int],
            next_tag: int):
        # Define shapes
        if side == 'L':
            if trans_x == notrans:
                w1_shape = x.value.shape[-ndim:] + add_shape
                w1_tile = x.value.basetile_shape[-ndim:] + add_basetile_shape

                interm_shape = x.value.shape[:-ndim] + add_shape
                interm_tile = x.value.basetile_shape[:-ndim] + add_basetile_shape

                w2_shape = add_shape + x.value.shape[-ndim:]
                w2_tile = add_basetile_shape + x.value.basetile_shape[-ndim:]

                y_shape = x.value.shape
                y_tile = x.value.basetile_shape
                # print("x shape: {}, w1 shape: {}, interm shape: {}, w2 shape: {}, y shape: {}".format(x.value.shape, w1_shape, interm_shape, w2_shape, y_shape))
            else:
                w1_shape = x.value.shape[:ndim] + add_shape
                w1_tile = x.value.basetile_shape[:ndim] + add_basetile_shape

                w2_shape = add_shape + x.value.shape[-ndim:]
                w2_tile = add_basetile_shape + x.value.basetile_shape[-ndim:]

                interm_shape = x.value.shape[:-ndim] + add_shape
                interm_tile = x.value.basetile_shape[:-ndim] + add_basetile_shape

                y_shape = x.value.shape
                y_tile = x.value.basetile_shape
                # print("x shape: {}, w1 shape: {}, interm shape: {}, w2 shape: {}, y shape: {}".format(x.value.shape, w1_shape, interm_shape, w2_shape, y_shape))

         
        if side == 'R':
            if trans_x == notrans:
                w1_shape = add_shape + x.value.shape[:ndim]
                w1_tile = add_basetile_shape + x.value.basetile_shape[:ndim] 

                interm_shape = add_shape + x.value.shape[ndim:]
                interm_tile = add_basetile_shape + x.value.basetile_shape[ndim:]

                w2_shape = x.value.shape[:ndim] + add_shape
                w2_tile = x.value.basetile_shape[:ndim] + add_basetile_shape

                y_shape = x.value.shape
                y_tile = x.value.basetile_shape
                # print("x shape: {}, w1 shape: {}, interm shape: {}, w2 shape: {}, y shape: {}".format(x.value.shape, w1_shape, interm_shape, w2_shape, y_shape))
            else:
                pass
        

        # Define W1
        w1_traits = TensorTraits(w1_shape, w1_tile)

        # Define W2
        w2_traits = TensorTraits(w2_shape, w2_tile)

        # Define y_interm
        interm1_traits = TensorTraits(interm_shape, interm_tile)
        interm2_traits = TensorTraits(interm_shape, interm_tile)

        # Define Y
        y_traits = TensorTraits(y_shape, y_tile)

        # TODO change distribution
        w1_distr = [0] * w1_traits.grid.nelems
        w1_value = type(x.value)(w1_traits, w1_distr, next_tag)
        next_tag = w1_value.next_tag

        # Create gradient of W1 with the same traits and distribution as W1
        w1_grad = type(x.value)(w1_traits, w1_distr, next_tag)
        next_tag = w1_grad.next_tag

        # TODO change distribution
        w2_distr = [0] * w2_traits.grid.nelems
        w2_value = type(x.value)(w2_traits, w2_distr, next_tag)
        next_tag = w2_value.next_tag

        # Create gradient of W2 with the same traits and distribution as W2
        w2_grad = type(x.value)(w2_traits, w2_distr, next_tag)
        next_tag = w2_grad.next_tag
        

        # TODO change distribution
        interm1_distr = [0] * interm1_traits.grid.nelems
        interm1_value = type(x.value)(interm1_traits, interm1_distr, next_tag)
        next_tag = interm1_value.next_tag

        # Create gradient of y_interm with the same traits and distribution as y_interm
        interm1_grad = type(x.value)(interm1_traits, interm1_distr, next_tag)
        next_tag = interm1_grad.next_tag

        # TODO change distribution
        interm2_distr = [0] * interm2_traits.grid.nelems
        interm2_value = type(x.value)(interm2_traits, interm2_distr, next_tag)
        next_tag = interm2_value.next_tag

        # Create gradient of y_interm with the same traits and distribution as y_interm
        interm2_grad = type(x.value)(interm2_traits, interm2_distr, next_tag)
        next_tag = interm2_grad.next_tag

        # Define W1 as TensorMoments
        w1 = TensorMoments(w1_value, w1_grad, True)

        # Define W2 as TensorMoments
        w2 = TensorMoments(w2_value, w2_grad, True)

        # Define y_interm as TensorMoments
        y_interm1 = TensorMoments(interm1_value, interm1_grad, True)
        y_interm2 = TensorMoments(interm2_value, interm2_grad, True)

        # TODO change distribution
        y_distr = [0] * y_traits.grid.nelems
        y_value = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_value.next_tag
        # Create gradient of Y with the same traits and distribution as Y
        y_grad = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_grad.next_tag

        # Define Y as TensorMoments
        y = TensorMoments(y_value, y_grad, True)

        # Bias is ignored for now
        #b = TensorMoments(None, None, False)
        # Create linear layer with all the provided data
        layer = MlpMixer(side, trans_x, x, y, w1, w2, y_interm1, y_interm2, ndim)
        # Return layer and next tag to be used
        return (layer, next_tag)


    # Forward propagation of the mlp_mixer layer
    def forward_async(self):
        # Perform actual gemm
        if self.side == 'L':
            gemm_async(1.0, self.trans_x, self.x.value, notrans, self.w1.value,
                    0.0, self.interm_1.value, self.ndim, 0)
            copy_async(self.interm_1.value, self.interm_2.value)
            gelu_async(self.interm_2.value)
            gemm_async(1.0, self.trans_x, self.interm_2.value, notrans, self.w2.value,
                    0.0, self.y.value, self.ndim, 0)
        else:
            gemm_async(1.0, notrans, self.w1.value, self.trans_x, self.x.value,
                    0.0, self.interm_1.value, self.ndim, 0)
            copy_async(self.interm_1.value, self.interm_2.value)
            gelu_async(self.interm_2.value)
            gemm_async(1.0, notrans, self.w2.value, self.trans_x, self.interm_2.value,
                    0.0, self.y.value, self.ndim, 0)
        # Hint for StarPU that W tensor will
        # not be used soon and it is advised to offload data from GPU
        self.w1.value.wont_use()
        self.w2.value.wont_use()