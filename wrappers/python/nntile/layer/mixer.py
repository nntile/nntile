# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/mixer.py
# Mixer layer of NNTile Python package
#
# @version 1.1.0

from nntile.layer.act import Act
from nntile.layer.base_layer import BaseLayer
from nntile.layer.layer_norm import LayerNorm
from nntile.layer.linear import Linear
from nntile.tensor import (
    TensorMoments, TensorTraits, add_async, add_slice_async, clear_async,
    notrans, sum_slice_async, transpose_async)


class GAP(BaseLayer):
    x: TensorMoments
    y: TensorMoments

    # Construct GAP layer with all the provided data
    def __init__(self, x: TensorMoments, y: TensorMoments, yT: TensorMoments):
        self.x = x
        self.y = y
        self.yT = yT
        # Redirect to BaseClass initialization
        super().__init__([x], [y], [], [yT])

    @staticmethod
    def generate_simple(x: TensorMoments, next_tag: int):
        yT_shape = x.value.shape[1:]
        yT_basetile_shape = x.value.basetile_shape[1:]
        yT_traits = TensorTraits(yT_shape, yT_basetile_shape)
        yT_distr = [0] * yT_traits.grid.nelems
        yT_value = type(x.value)(yT_traits, yT_distr, next_tag)
        next_tag = yT_value.next_tag

        # Create gradient of Y with the same traits and distribution as Y
        yT_grad = type(x.value)(yT_traits, yT_distr, next_tag)
        next_tag = yT_grad.next_tag
        # Define Y as TensorMoments
        yT = TensorMoments(yT_value, yT_grad, True)

        y_shape = yT_shape[::-1]
        y_basetile_shape = yT_basetile_shape[::-1]
        y_traits = TensorTraits(y_shape, y_basetile_shape)
        y_distr = [0] * y_traits.grid.nelems
        y_value = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_value.next_tag

        # Create gradient of Y with the same traits and distribution as Y
        y_grad = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_grad.next_tag
        # Define Y as TensorMoments
        y = TensorMoments(y_value, y_grad, True)

        # Create GAP layer with all the provided data
        layer = GAP(x, y, yT)
        # Return layer and next tag to be used
        return (layer, next_tag)

    def forward_async(self):
        alpha = 1 / (self.x.value.shape[0])
        sum_slice_async(alpha, self.x.value, 0.0, self.yT.value, 0)
        transpose_async(1.0, self.yT.value, self.y.value, 1)

    def backward_async(self):
        alpha = 1 / (self.x.value.shape[0])
        transpose_async(1.0, self.y.grad, self.yT.grad, 1)
        add_slice_async(alpha, self.yT.grad, 0.0, self.x.grad, 0)


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

        # Set up local named parameters
        self.side = side
        self.x = x
        self.y = y
        self.linear_1 = linear_1
        self.linear_2 = linear_2
        self.act = act
        layer_temporaries = list(self.linear_1.activations_output +
                self.act.activations_output + self.linear_2.activations_output)
        layer_parameters = list(self.linear_1.parameters +
                self.linear_2.parameters)
        # Redirect to BaseClass initialization
        super().__init__([x], [y], layer_parameters, layer_temporaries)

    # Clear gradients of activations and parameters
    def clear_gradients(self):
        for t in (self.activations_input + self.activations_output +
                self.temporaries):
            if t.grad is not None and t.grad_required:
                clear_async(t.grad)
        for t in self.parameters:
            if t.grad is not None and t.grad_required:
                clear_async(t.grad)

    @staticmethod
    def generate_simple(x: TensorMoments, side: str, next_tag: int):
        if side == 'R':
            add_shape = x.value.shape[0] * 4
            add_basetile_shape = x.value.shape[0] * 4
            init_shape = x.value.shape[0]
        if side == 'L':
            add_shape = x.value.shape[-1] * 4
            add_basetile_shape = x.value.shape[-1] * 4
            init_shape = x.value.shape[-1]
        linear_1_layer, next_tag = Linear.generate_simple(x, side, notrans, 1,
                [add_shape], [add_basetile_shape], next_tag, bias=False)
        act_layer, next_tag = Act.generate_simple(linear_1_layer.y, 'gelu',
                next_tag)
        linear_2_layer, next_tag = Linear.generate_simple(act_layer.y, side,
                notrans, 1, [init_shape], [init_shape], next_tag, bias=False)
        layer = MixerMlp(side, x, linear_2_layer.y, linear_1_layer,
                linear_2_layer, act_layer)
        # Return layer and next tag to be used
        return (layer, next_tag)

    def forward_async(self):
        self.linear_1.forward_async()
        self.act.forward_async()
        self.linear_2.forward_async()

    def backward_async(self):
        self.linear_2.backward_async()
        self.act.backward_async()
        self.linear_1.backward_async()


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

        # Set up local named parameters
        self.x = x
        self.y = y
        self.norm_1 = norm_1
        self.norm_2 = norm_2
        self.mlp_1 = mlp_1
        self.mlp_2 = mlp_2
        layer_parameters = list(self.norm_1.parameters +
                self.mlp_1.parameters + self.norm_2.parameters +
                self.mlp_2.parameters)
        layer_tmp = list(self.norm_1.activations_output +
                self.mlp_1.temporaries + self.norm_2.activations_output +
                self.mlp_2.temporaries)
        # Redirect to BaseClass initialization
        super().__init__([x], [y], layer_parameters, layer_tmp)

    def unregister(self):
        super().unregister()
        self.norm_1.unregister()
        self.norm_2.unregister()
        self.mlp_1 .unregister()
        self.mlp_2.unregister()

    # Clear gradients of activations and parameters
    def clear_gradients(self):
        for t in (self.activations_input + self.activations_output
                + self.temporaries):
            if t.grad is not None and t.grad_required:
                clear_async(t.grad)
        for t in self.parameters:
            if t.grad is not None and t.grad_required:
                clear_async(t.grad)

    # Simple generator for the mixer layer
    @staticmethod
    def generate_simple(x: TensorMoments, next_tag: int):
        eps = 1e-5
        norm_1_layer, next_tag = LayerNorm.generate_simple(x, 2, eps, next_tag)
        mlp_layer1, next_tag = MixerMlp.generate_simple(norm_1_layer.y, 'R',
                next_tag)
        norm_2_layer, next_tag = LayerNorm.generate_simple(mlp_layer1.y, 2,
                eps, next_tag)
        mlp_layer2, next_tag = MixerMlp.generate_simple(norm_2_layer.y, 'L',
                next_tag)
        layer = Mixer(x, mlp_layer2.y, norm_1_layer, norm_2_layer, mlp_layer1,
                mlp_layer2)
        return layer, next_tag

    # Forward propagation of the mixer layer
    def forward_async(self):
        self.norm_1.forward_async()
        self.mlp_1.forward_async()
        add_async(1.0, self.x.value, 1.0, self.mlp_1.y.value)
        self.norm_2.forward_async()
        self.mlp_2.forward_async()
        add_async(1.0, self.norm_2.x.value, 1.0, self.mlp_2.y.value)

    def backward_async(self):
        self.mlp_2.backward_async()
        self.norm_2.backward_async()
        add_async(1.0, self.mlp_2.linear_2.y.grad, 1.0, self.mlp_1.y.grad)
        self.mlp_1.backward_async()
        self.norm_1.backward_async()
        add_async(1.0, self.mlp_1.linear_2.y.grad, 1.0, self.x.grad)
