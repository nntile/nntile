/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/hypot_scalar_inverse.cc
 * TileGraph hypot scalar inverse operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/hypot_scalar_inverse.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/hypot_scalar_inverse.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(TileGraph::Runtime& runtime, Scalar e, Scalar a, TileGraph::TileNode* d)
{
    nntile::tile::hypot_scalar_inverse<T>(e, a, runtime.get_tile<T>(d));
}
} // namespace
void hypot_scalar_inverse(Scalar eps, Scalar alpha, TileGraph::TileNode* dst)
{
    if(!dst)
        throw std::invalid_argument("tile hypot_scalar_inverse: null");
    dst->graph()->add_op(std::make_shared<TileHypotScalarInverseOp>(eps, alpha, dst));
}
void TileHypotScalarInverseOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dst);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, eps, alpha, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, eps, alpha, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, eps, alpha, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, eps, alpha, dst);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, eps, alpha, dst);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, eps, alpha, dst);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, eps, alpha, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) + " not supported for hypot_scalar_inverse");
        default:
            throw std::runtime_error("Unsupported data type for hypot_scalar_inverse");
    }
}
} // namespace nntile::graph::tile_graph
