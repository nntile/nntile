/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_broadcast.cpp
 * Broadcast and repeat via chained ``scale_slice`` (one axis per op).
 */

#include "nntile_broadcast.h"

#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_context.h"
#include "nntile_tensor_gc.h"

#include <stdexcept>

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <ATen/Tensor.h>
#include <nntile/tensor/ops/scale_slice.hh>
#include <nntile/tensor/ops/clear.hh>

#include <cstring>
#include <stdexcept>
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

void insert_axis_size(
    const std::vector<nntile::Index> &shape,
    nntile::Index axis,
    nntile::Index axis_size,
    std::vector<nntile::Index> &out)
{
    out.clear();
    out.reserve(shape.size() + 1);
    for (nntile::Index i = 0; i < axis; ++i)
    {
        out.push_back(shape[static_cast<std::size_t>(i)]);
    }
    out.push_back(axis_size);
    for (nntile::Index i = axis; i < static_cast<nntile::Index>(shape.size());
         ++i)
    {
        out.push_back(shape[static_cast<std::size_t>(i)]);
    }
}

} // namespace

nntile::TensorGraph::TensorNode *broadcast_scale_slice_chain(
    nntile::TensorGraph::TensorNode *src,
    nntile::TensorGraph::TensorNode *dst,
    nntile::TensorGraph &graph,
    const std::vector<nntile::Index> &dst_shape)
{
    if (src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "broadcast_scale_slice_chain: src and dst must be non-null");
    }
    if (dst_shape.empty())
    {
        return src;
    }

    nntile::TensorGraph::TensorNode *src_node = src;
    for (std::size_t dim = 0; dim < dst_shape.size(); ++dim)
    {
        nntile::TensorGraph::TensorNode *dst_node = dst;
        if (dim + 1 < dst_shape.size())
        {
            std::vector<nntile::Index> partial_shape(
                dst_shape.begin(),
                dst_shape.begin() + static_cast<std::ptrdiff_t>(dim) + 1);
            dst_node = graph.data(partial_shape, src->dtype())
                           ->set_name("broadcast_scale_slice");
            track_graph_node(dst_node);
        }
        nntile::tensor::scale_slice(
            static_cast<nntile::Scalar>(1.0),
            src_node,
            dst_node,
            static_cast<nntile::Index>(dim));
        src_node = dst_node;
    }
    return src_node;
}

nntile::TensorGraph::TensorNode *repeat_scale_slice_chain(
    nntile::TensorGraph::TensorNode *src,
    const std::vector<nntile::Index> &input_shape,
    const std::vector<nntile::Index> &repeats,
    nntile::TensorGraph &graph,
    nntile::DataType dtype,
    const at::Tensor &out,
    const std::vector<nntile::Index> &out_shape,
    std::vector<nntile::Index> &graph_shape_out)
{
    const std::size_t input_ndim = input_shape.size();
    const std::size_t repeat_ndim = repeats.size();
    if (repeat_ndim < input_ndim)
    {
        throw std::invalid_argument(
            "repeat_scale_slice_chain: repeats must have at least input ndim");
    }

    const std::size_t pad = repeat_ndim - input_ndim;
    nntile::TensorGraph::TensorNode *cur = src;
    std::vector<nntile::Index> cur_graph_shape = input_shape;
    std::size_t inserted = 0;
    bool recorded = false;

    std::size_t remaining = 0;
    for (std::size_t d = 0; d < repeat_ndim; ++d)
    {
        if (repeats[d] != 1 || d < pad)
        {
            ++remaining;
        }
    }

    std::size_t step = 0;
    for (std::size_t d = 0; d < repeat_ndim; ++d)
    {
        const nntile::Index repeat_factor = repeats[d];
        const bool needs_axis = repeat_factor != 1 || d < pad;
        if (!needs_axis)
        {
            continue;
        }

        nntile::Index graph_axis = 0;
        if (d >= pad)
        {
            graph_axis = static_cast<nntile::Index>(d + inserted - pad);
        }

        std::vector<nntile::Index> next_graph_shape;
        insert_axis_size(cur_graph_shape, graph_axis, repeat_factor, next_graph_shape);

        ++step;
        const bool is_last = step == remaining;

        nntile::TensorGraph::TensorNode *dst_node = nullptr;
        if (is_last)
        {
            dst_node = get_or_create_data_node(
                out,
                next_graph_shape,
                dtype,
                false);
            graph_shape_out = next_graph_shape;
        }
        else
        {
            dst_node = graph.data(next_graph_shape, dtype)
                           ->set_name("repeat_scale_slice");
            track_graph_node(dst_node);
        }

        nntile::tensor::scale_slice(
            static_cast<nntile::Scalar>(1.0),
            cur,
            dst_node,
            graph_axis);
        cur = dst_node;
        cur_graph_shape = std::move(next_graph_shape);
        ++inserted;
        recorded = true;
    }

    if (!recorded)
    {
        graph_shape_out = input_shape;
        if (out_shape != input_shape)
        {
            throw std::invalid_argument(
                "repeat_scale_slice_chain: no ops recorded for shape change");
        }
        return src;
    }

    (void)out_shape;
    return cur;
}

void tensor_repeat_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    c10::IntArrayRef repeats)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input.sizes());
    const std::vector<nntile::Index> repeats_graph =
        pytorch_shape_to_graph(repeats);
    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out.sizes());

    auto *src_node = get_or_create_data_node(
        input,
        input_graph,
        nntile::DataType::FP32,
        true);

    std::vector<nntile::Index> graph_shape;
    auto *out_node = repeat_scale_slice_chain(
        src_node,
        input_graph,
        repeats_graph,
        *src_node->graph(),
        nntile::DataType::FP32,
        out,
        out_graph,
        graph_shape);

    if (out_node == src_node)
    {
        if (has_host_staging(input) && has_host_staging(out))
        {
            std::size_t count = 1;
            for (const nntile::Index dim : out_graph)
            {
                count *= static_cast<std::size_t>(dim);
            }
            if (count > 0)
            {
                sync_runtime_to_nntile_tensor(input);
                std::memcpy(
                    out.data_ptr<float>(),
                    input.data_ptr<float>(),
                    count * sizeof(float));
            }
        }
        else
        {
            register_data_node(out, src_node);
            maybe_execute_after_record();
        }
        return;
    }

    register_data_node(out, out_node);
    maybe_execute_after_record();
}

void tensor_broadcast_scalar_fp32(
    const at::Tensor &scalar,
    at::Tensor &out)
{
    const std::vector<nntile::Index> dst_graph =
        pytorch_shape_to_graph(out.sizes());
    if (dst_graph.empty())
    {
        if (scalar.data_ptr<float>() != out.data_ptr<float>())
        {
            std::memcpy(
                out.data_ptr<float>(),
                scalar.data_ptr<float>(),
                sizeof(float));
        }
        return;
    }

    auto *src_node = get_or_create_data_node(
        scalar,
        std::vector<nntile::Index>{},
        nntile::DataType::FP32,
        true);
    auto *dst_node = get_or_create_data_node(
        out,
        dst_graph,
        nntile::DataType::FP32,
        false);
    nntile::tensor::clear(dst_node);
    broadcast_scale_slice_chain(
        src_node,
        dst_node,
        *src_node->graph(),
        dst_graph);
    register_data_node(out, dst_node);
    maybe_execute_after_record();
}

} // namespace torch_nntile

#else

namespace torch_nntile
{

void tensor_repeat_fp32(
    const at::Tensor & /*input*/,
    at::Tensor & /*out*/,
    c10::IntArrayRef /*repeats*/)
{
    throw std::runtime_error("tensor_repeat_fp32 requires libnntile");
}

void tensor_broadcast_scalar_fp32(
    const at::Tensor & /*scalar*/,
    at::Tensor & /*out*/)
{
    throw std::runtime_error(
        "tensor_broadcast_scalar_fp32 requires libnntile");
}

} // namespace torch_nntile

#endif // TORCH_NNTILE_USE_LIBNNTILE
