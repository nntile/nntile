#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tile_graph/multiply_fiber_inplace.cc
 * TileGraph multiply fiber inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tile/ops/multiply_fiber_inplace.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/dtype.hh>
#include <nntile/core.hh>
#include <nntile/core/multiply_fiber_inplace.hh>

#include <nntile/runtime.hh>
#include <nntile/tile/shape_layout.hh>
namespace nntile::tile
{
namespace
{
template<typename T>
void run(Runtime& rt, Scalar a, TileGraph::TileNode* s, TileGraph::TileNode* d, Index ax)
{
    nntile::core::multiply_fiber_inplace<T>(rt.starpu_worker_hint(), a, rt.get_tile<T>(s), rt.get_tile<T>(d), ax);
}
} // namespace
void multiply_fiber_inplace(Scalar a, TileGraph::TileNode* s, TileGraph::TileNode* d, Index axis)
{
    if(!s || !d)
        throw std::invalid_argument("multiply_fiber_inplace");
    if(s->graph() != d->graph() || s->dtype() != d->dtype() || s == d)
        throw std::invalid_argument("multiply_fiber_inplace");
    s->graph()->add_op(std::make_shared<TileMultiplyFiberInplaceOp>(a, s, d, axis));
}
void TileMultiplyFiberInplaceOp::execute(Runtime& runtime) const
{
    const Index s_axis =
        tensor::graph_axis_to_storage(axis, dst->ndim());
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, alpha, src, dst, s_axis);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, alpha, src, dst, s_axis);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, alpha, src, dst, s_axis);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, alpha, src, dst, s_axis);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, alpha, src, dst, s_axis);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, alpha, src, dst, s_axis);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, alpha, src, dst, s_axis);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("multiply_fiber_inplace");
        default:
            throw std::runtime_error("multiply_fiber_inplace");
    }
}
} // namespace nntile::tile
