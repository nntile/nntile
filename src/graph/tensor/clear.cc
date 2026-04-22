/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/clear.cc
 * TensorGraph clear operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/clear.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/clear.hh"

#include <nntile/graph/tile/graph_ops.hh>
#include <nntile/graph/tensor/tile_lowering_helpers.hh>

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_clear(
    TensorGraph::Runtime& runtime,
    TensorGraph::TensorNode* x)
{
    auto& x_t = runtime.get_tensor<T>(x);
    nntile::tensor::clear<T>(x_t);
}

} // namespace

void clear(TensorGraph::TensorNode* x)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("clear: input tensor must be non-null");
    }

    auto op = std::make_shared<TensorClearOp>(x);
    x->graph()->add_op(op);
}

void TensorClearOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_clear<nntile::fp32_t>(runtime, x);
            break;
        case DataType::FP32_FAST_TF32:
            run_clear<nntile::fp32_fast_tf32_t>(runtime, x);
            break;
        case DataType::FP32_FAST_FP16:
            run_clear<nntile::fp32_fast_fp16_t>(runtime, x);
            break;
        case DataType::FP32_FAST_BF16:
            run_clear<nntile::fp32_fast_bf16_t>(runtime, x);
            break;
        case DataType::FP64:
            run_clear<nntile::fp64_t>(runtime, x);
            break;
        case DataType::FP16:
            run_clear<nntile::fp16_t>(runtime, x);
            break;
        case DataType::BF16:
            run_clear<nntile::bf16_t>(runtime, x);
            break;
        case DataType::INT64:
            run_clear<nntile::int64_t>(runtime, x);
            break;
        case DataType::BOOL:
            run_clear<nntile::bool_t>(runtime, x);
            break;
        default:
            throw std::runtime_error("Unsupported data type for clear");
    }
}

void TensorClearOp::lower_to_tile(const LoweringContext& ctx) const
{
    for(TileGraph::TileNode* t : tile_lower::tiles_of(ctx.tile_map, x))
    {
        tile_graph::clear(t);
    }
}

} // namespace nntile::graph::tensor
