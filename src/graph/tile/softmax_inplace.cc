/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/softmax_inplace.cc
 * TileGraph softmax inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/softmax_inplace.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/softmax_inplace.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run_smi(
    TileGraph::Runtime& runtime, TileGraph::TileNode* m, Scalar a, TileGraph::TileNode* d, Index ax)
{
    nntile::tile::softmax_inplace<T>(runtime.get_tile<T>(m), a, runtime.get_tile<T>(d), ax);
}
} // namespace
void softmax_inplace(TileGraph::TileNode* mse, Scalar alpha, TileGraph::TileNode* dst, Index axis)
{
    if(!mse || !dst)
        throw std::invalid_argument("tile softmax_inplace: null");
    if(mse->graph() != dst->graph())
        throw std::invalid_argument("tile softmax_inplace: same graph");
    if(mse->dtype() != dst->dtype())
        throw std::invalid_argument("tile softmax_inplace: dtype");
    if(mse == dst)
        throw std::invalid_argument("tile softmax_inplace: maxsumexp and dst must be distinct");
    // Shape compatibility follows nntile::tile::softmax_inplace
    mse->graph()->add_op(std::make_shared<TileSoftmaxInplaceOp>(mse, alpha, dst, axis));
}
void TileSoftmaxInplaceOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dst);
    switch(dtype)
    {
        case DataType::FP32:
            run_smi<nntile::fp32_t>(runtime, maxsumexp_n, alpha, dst, axis);
            break;
        case DataType::FP32_FAST_TF32:
            run_smi<nntile::fp32_fast_tf32_t>(runtime, maxsumexp_n, alpha, dst, axis);
            break;
        case DataType::FP32_FAST_FP16:
            run_smi<nntile::fp32_fast_fp16_t>(runtime, maxsumexp_n, alpha, dst, axis);
            break;
        case DataType::FP32_FAST_BF16:
            run_smi<nntile::fp32_fast_bf16_t>(runtime, maxsumexp_n, alpha, dst, axis);
            break;
        case DataType::FP64:
            run_smi<nntile::fp64_t>(runtime, maxsumexp_n, alpha, dst, axis);
            break;
        case DataType::FP16:
            run_smi<nntile::fp16_t>(runtime, maxsumexp_n, alpha, dst, axis);
            break;
        case DataType::BF16:
            run_smi<nntile::bf16_t>(runtime, maxsumexp_n, alpha, dst, axis);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) + " not supported for softmax_inplace");
        default:
            throw std::runtime_error("Unsupported data type for softmax_inplace");
    }
}
} // namespace nntile::graph::tile_graph
