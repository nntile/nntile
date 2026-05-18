/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/mask_scalar.cc
 * TileGraph mask scalar operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/mask_scalar.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/mask_scalar.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run_mask(
    TileGraph::Runtime& runtime, TileGraph::TileNode* m, Scalar v, TileGraph::TileNode* A, Index bnd)
{
    nntile::tile::mask_scalar<T>(runtime.get_tile<nntile::bool_t>(m), v, runtime.get_tile<T>(A), bnd);
}
} // namespace
void mask_scalar(TileGraph::TileNode* mask, Scalar val, TileGraph::TileNode* a, Index batch_ndim)
{
    if(!mask || !a)
        throw std::invalid_argument("tile mask_scalar: null");
    if(mask->graph() != a->graph())
        throw std::invalid_argument("tile mask_scalar: same graph");
    if(mask->dtype() != DataType::BOOL)
        throw std::invalid_argument("tile mask_scalar: mask must be BOOL");
    a->graph()->add_op(std::make_shared<TileMaskScalarOp>(mask, val, a, batch_ndim));
}
void TileMaskScalarOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(a);
    switch(dtype)
    {
        case DataType::FP32:
            run_mask<nntile::fp32_t>(runtime, mask, val, a, batch_ndim);
            break;
        case DataType::FP32_FAST_TF32:
            run_mask<nntile::fp32_fast_tf32_t>(runtime, mask, val, a, batch_ndim);
            break;
        case DataType::FP32_FAST_FP16:
            run_mask<nntile::fp32_fast_fp16_t>(runtime, mask, val, a, batch_ndim);
            break;
        case DataType::FP32_FAST_BF16:
            run_mask<nntile::fp32_fast_bf16_t>(runtime, mask, val, a, batch_ndim);
            break;
        case DataType::FP64:
            run_mask<nntile::fp64_t>(runtime, mask, val, a, batch_ndim);
            break;
        case DataType::FP16:
            run_mask<nntile::fp16_t>(runtime, mask, val, a, batch_ndim);
            break;
        case DataType::BF16:
            run_mask<nntile::bf16_t>(runtime, mask, val, a, batch_ndim);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("INT64/BOOL not supported for mask_scalar data tile");
        default:
            throw std::runtime_error("Unsupported data type for mask_scalar");
    }
}
} // namespace nntile::graph::tile_graph
