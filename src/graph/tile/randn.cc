/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/randn.cc
 * TileGraph randn operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/randn.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/randn.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run_rd(
    TileGraph::Runtime& runtime, TileGraph::TileNode* d, const std::vector<Index>& st, const std::vector<Index>& us, unsigned long long sd, Scalar m, Scalar s)
{
    nntile::tile::randn<T>(runtime.get_tile<T>(d), st, us, sd, m, s);
}
} // namespace
void randn(
    TileGraph::TileNode* dst,
    const std::vector<Index>& start,
    const std::vector<Index>& underlying_shape,
    unsigned long long seed,
    Scalar mean,
    Scalar stddev)
{
    if(!dst)
        throw std::invalid_argument("tile randn: null");
    dst->graph()->add_op(
        std::make_shared<TileRandnOp>(start, underlying_shape, seed, mean, stddev, dst));
}
void TileRandnOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dst);
    switch(dtype)
    {
        case DataType::FP32:
            run_rd<nntile::fp32_t>(runtime, dst, start, underlying_shape, seed, mean, stddev);
            break;
        case DataType::FP32_FAST_TF32:
            run_rd<nntile::fp32_fast_tf32_t>(runtime, dst, start, underlying_shape, seed, mean, stddev);
            break;
        case DataType::FP32_FAST_FP16:
            run_rd<nntile::fp32_fast_fp16_t>(runtime, dst, start, underlying_shape, seed, mean, stddev);
            break;
        case DataType::FP32_FAST_BF16:
            run_rd<nntile::fp32_fast_bf16_t>(runtime, dst, start, underlying_shape, seed, mean, stddev);
            break;
        case DataType::FP64:
            run_rd<nntile::fp64_t>(runtime, dst, start, underlying_shape, seed, mean, stddev);
            break;
        case DataType::FP16:
            throw std::runtime_error("FP16 not supported for tile randn in this build");
        case DataType::BF16:
            run_rd<nntile::bf16_t>(runtime, dst, start, underlying_shape, seed, mean, stddev);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) + " not supported for randn");
        default:
            throw std::runtime_error("Unsupported data type for randn");
    }
}
} // namespace nntile::graph::tile_graph
