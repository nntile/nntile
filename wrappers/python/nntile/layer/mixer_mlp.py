# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/mixer_mlp.py
# Token-mixing / channel-mixing MLP submodule of MLP-Mixer
#
# @version 1.1.0

from nntile.layer.act import Act
from nntile.layer.base_layer import BaseLayer
from nntile.layer.linear import Linear
from nntile.tensor import TensorMoments, clear_async, notrans


class MixerMlp(BaseLayer):
    side: str
    x: TensorMoments
    y: TensorMoments
    linear_1: Linear
    linear_2: Linear
    act: Act

    def __init__(
        self,
        side: str,
        x: TensorMoments,
        y: TensorMoments,
        linear_1: Linear,
        linear_2: Linear,
        act: Act,
    ):
        if side not in ('L', 'R'):
            raise ValueError("side must be either 'L' or 'R'")

        self.side = side
        self.x = x
        self.y = y
        self.linear_1 = linear_1
        self.linear_2 = linear_2
        self.act = act
        layer_temporaries = list(
            self.linear_1.activations_output + self.act.activations_output
        )
        layer_parameters = (
            self.linear_1.parameters + self.linear_2.parameters
        )
        super().__init__([x], [y], layer_parameters, layer_temporaries)

    def clear_gradients(self):
        for tensor in (
            self.activations_input
            + self.activations_output
            + self.temporaries
        ):
            if tensor.grad is not None and tensor.grad_required:
                clear_async(tensor.grad)
        for tensor in self.parameters:
            if tensor.grad is not None and tensor.grad_required:
                clear_async(tensor.grad)

    @staticmethod
    def generate_simple(x: TensorMoments, side: str):
        if side not in ('L', 'R'):
            raise ValueError("side must be either 'L' or 'R'")
        if side == 'R':
            add_shape = x.value.shape[0] * 4
            add_basetile_shape = x.value.basetile_shape[0] * 4
            init_shape = x.value.shape[0]
            init_basetile_shape = x.value.basetile_shape[0]
        else:
            add_shape = x.value.shape[-1] * 4
            add_basetile_shape = x.value.basetile_shape[-1] * 4
            init_shape = x.value.shape[-1]
            init_basetile_shape = x.value.basetile_shape[-1]

        linear_1_layer = Linear.generate_simple(
            x, side, notrans, 1,
            [add_shape], [add_basetile_shape], bias=False,
        )
        act_layer = Act.generate_simple(linear_1_layer.y, 'gelu')
        linear_2_layer = Linear.generate_simple(
            act_layer.y, side, notrans, 1,
            [init_shape], [init_basetile_shape], bias=False,
        )
        return MixerMlp(
            side, x, linear_2_layer.y, linear_1_layer,
            linear_2_layer, act_layer,
        )

    def forward_async(self):
        self.linear_1.forward_async()
        self.act.forward_async()
        self.linear_2.forward_async()

    def backward_async(self):
        self.linear_2.backward_async()
        self.act.backward_async()
        self.linear_1.backward_async()
