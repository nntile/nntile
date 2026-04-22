/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/multiply_inplace.cc
 * TensorGraph multiply_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/multiply_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/multiply_inplace.hh"
#include "nntile/tensor/multiply_inplace.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_multiply_inplace(
    TensorGraph::Runtime& runtime,
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::multiply_inplace<T>(alpha, src_t, dst_t);
}

} // namespace

void multiply_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must have the same dtype");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "multiply_inplace: src and dst must be distinct tensors");
    }
    validate_same_shape_and_merge(src, dst, "multiply_inplace");

    auto op = std::make_shared<TensorMultiplyInplaceOp>(src, dst, alpha);
    src->graph()->add_op(op);
}

void TensorMultiplyInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_multiply_inplace<nntile::fp32_t>(runtime, alpha, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_multiply_inplace<nntile::fp32_fast_tf32_t>(runtime, alpha, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_multiply_inplace<nntile::fp32_fast_fp16_t>(runtime, alpha, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_multiply_inplace<nntile::fp32_fast_bf16_t>(runtime, alpha, src, dst);
            break;
        case DataType::FP64:
            run_multiply_inplace<nntile::fp64_t>(runtime, alpha, src, dst);
            break;
        case DataType::FP16:
            run_multiply_inplace<nntile::fp16_t>(runtime, alpha, src, dst);
            break;
        case DataType::BF16:
            run_multiply_inplace<nntile::bf16_t>(runtime, alpha, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for multiply_inplace");
        default:
            throw std::runtime_error("Unsupported data type for multiply_inplace");
    }
}

void TensorMultiplyInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& tiles_src = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_dst = tile_lower::tiles_of(ctx.tile_map, dst);
    if(tiles_src.size() != tiles_dst.size())
    {
        throw std::runtime_error(
            "lower_to_tile MULTIPLY_INPLACE: tile count mismatch");
    }
    tile_lower::assert_same_elementwise_layout(
        src, dst, "MULTIPLY_INPLACE src/dst");
    for(size_t i = 0; i < tiles_src.size(); ++i)
    {
        tile_graph::multiply_inplace(alpha, tiles_src[i], tiles_dst[i]);
    }
}

} // namespace nntile::graph::tensor
