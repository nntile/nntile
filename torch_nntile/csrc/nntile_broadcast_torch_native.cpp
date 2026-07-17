/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_broadcast_torch_native.cpp
 * Repeat / broadcast helpers via torch-native StarPU path.
 */

#include "nntile_broadcast.h"

#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_meta.h"

#include <c10/util/Exception.h>

#include <nntile/tensor/ops/torch_dispatch.hh>

#include <vector>

namespace torch_nntile
{

namespace
{

std::vector<nntile::Index> pytorch_shape_to_graph(c10::IntArrayRef shape)
{
    std::vector<nntile::Index> graph_shape;
    graph_shape.reserve(shape.size());
    for (const auto dim : shape)
    {
        graph_shape.push_back(static_cast<nntile::Index>(dim));
    }
    return graph_shape;
}

bool mark_as_input_for_operand(const at::Tensor &tensor)
{
    return tensor.device().is_cpu();
}

} // namespace

void tensor_repeat_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    c10::IntArrayRef repeats)
{
    TORCH_CHECK(
        repeats.size() == static_cast<size_t>(input.dim()) ||
            repeats.size() >= static_cast<size_t>(input.dim()),
        "tensor_repeat_fp32: bad repeats rank");
    const auto in_shape = pytorch_shape_to_graph(input.sizes());
    const auto out_shape = pytorch_shape_to_graph(out.sizes());
    auto *in_node = get_or_create_data_node(
        input,
        in_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    nntile::starpu::TorchDispatchArgs extra{};
    // Pad leading dims: aten repeat may have more entries than input rank.
    const std::size_t pad =
        repeats.size() > static_cast<std::size_t>(input.dim())
            ? repeats.size() - static_cast<std::size_t>(input.dim())
            : 0;
    // Codelet expects repeats aligned to the *input* tensor rank after
    // leading 1-padding has been materialized into out_shape. Pass the
    // trailing input.dim() repeat factors when ranks match out; otherwise
    // pass full repeats for the output rank via a view-less copy path.
    const nntile::Index ndim =
        static_cast<nntile::Index>(out.dim());
    TORCH_CHECK(
        ndim <= nntile::starpu::torch_dispatch_max_ndim,
        "tensor_repeat_fp32: ndim too large");
    std::vector<std::int64_t> full_repeats(
        static_cast<size_t>(ndim),
        1);
    for (std::size_t i = 0; i < repeats.size(); ++i)
    {
        full_repeats[i] = repeats[static_cast<std::int64_t>(i)];
    }
    TORCH_CHECK(
        pad == 0 ||
            input.dim() + static_cast<std::int64_t>(pad) == out.dim(),
        "tensor_repeat_fp32: padded rank mismatch");
    at::Tensor src = input;
    if (pad > 0)
    {
        std::vector<std::int64_t> view_shape(
            static_cast<size_t>(pad),
            1);
        for (const std::int64_t d : input.sizes())
        {
            view_shape.push_back(d);
        }
        src = input.reshape(view_shape);
    }
    const auto src_shape = pytorch_shape_to_graph(src.sizes());
    auto *src_node = get_or_create_data_node(
        src,
        src_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(src));
    for (nntile::Index i = 0; i < ndim; ++i)
    {
        extra.iargs[i] = static_cast<nntile::Index>(
            full_repeats[static_cast<size_t>(i)]);
    }
    auto *out_node = nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::Repeat,
        src_node,
        out_shape,
        extra);
    register_data_node(out, out_node);
    (void)in_node;
}

void tensor_broadcast_scalar_fp32(
    const at::Tensor &scalar,
    at::Tensor &out)
{
    // out = scalar.expand_as(out) via repeat factors.
    TORCH_CHECK(scalar.numel() == 1, "broadcast_scalar: expected numel=1");
    std::vector<std::int64_t> repeats;
    repeats.reserve(static_cast<size_t>(out.dim()));
    for (const std::int64_t d : out.sizes())
    {
        repeats.push_back(d);
    }
    at::Tensor src = scalar.reshape(std::vector<std::int64_t>(
        static_cast<size_t>(out.dim()),
        1));
    tensor_repeat_fp32(src, out, repeats);
}

nntile::TensorGraph::TensorNode *broadcast_scale_slice_chain(
    nntile::TensorGraph::TensorNode *,
    nntile::TensorGraph::TensorNode *,
    nntile::TensorGraph &,
    const std::vector<nntile::Index> &)
{
    TORCH_CHECK(
        false,
        "broadcast_scale_slice_chain disabled under "
        "NNTILE_TORCH_NATIVE_OPS");
}

} // namespace torch_nntile
