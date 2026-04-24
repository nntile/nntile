/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/log_scalar.cc
 * TileGraph log scalar operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/log_scalar.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/log_scalar.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run_ls(TileGraph::Runtime& runtime, const std::string& n, TileGraph::TileNode* v)
{
    nntile::tile::log_scalar<T>(n, runtime.get_tile<T>(v));
}
} // namespace
void log_scalar(const std::string& name, TileGraph::TileNode* value)
{
    if(!value)
        throw std::invalid_argument("tile log_scalar: null");
    value->graph()->add_op(std::make_shared<TileLogScalarOp>(name, value));
}
void TileLogScalarOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(value);
    switch(dtype)
    {
        case DataType::FP32:
            run_ls<nntile::fp32_t>(runtime, name, value);
            break;
        case DataType::FP32_FAST_TF32:
            run_ls<nntile::fp32_fast_tf32_t>(runtime, name, value);
            break;
        case DataType::FP32_FAST_FP16:
            run_ls<nntile::fp32_fast_fp16_t>(runtime, name, value);
            break;
        case DataType::FP32_FAST_BF16:
            run_ls<nntile::fp32_fast_bf16_t>(runtime, name, value);
            break;
        case DataType::FP64:
            run_ls<nntile::fp64_t>(runtime, name, value);
            break;
        case DataType::FP16:
            throw std::runtime_error("FP16 not supported for tile log_scalar in this build");
        case DataType::BF16:
            run_ls<nntile::bf16_t>(runtime, name, value);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) + " not supported for log_scalar");
        default:
            throw std::runtime_error("Unsupported data type for log_scalar");
    }
}
} // namespace nntile::graph::tile_graph
