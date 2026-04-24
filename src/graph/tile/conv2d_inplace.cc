/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/conv2d_inplace.cc
 * TileGraph conv2d inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/conv2d_inplace.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/conv2d_inplace.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(const TileConv2dInplaceOp& o, TileGraph::Runtime& runtime)
{
    nntile::tile::conv2d_inplace<T>(o.src1_m, o.src1_n, o.src1_channels, o.batch, o.src2_m, o.src2_n, o.dilation_m, o.dilation_n, o.dst_channels, o.offset_m, o.offset_n, o.alpha, runtime.get_tile<T>(o.s1), runtime.get_tile<T>(o.s2), o.dst_m, o.dst_n, o.stride_m, o.stride_n, o.beta, runtime.get_tile<T>(o.dst));
}
} // namespace
void conv2d_inplace(
    Index src1_m, Index src1_n, Index src1_channels, Index batch, Index src2_m, Index src2_n, Index dilation_m, Index dilation_n, Index dst_channels, Index offset_m, Index offset_n, Scalar alpha, TileGraph::TileNode* src1, TileGraph::TileNode* src2, Index dst_m, Index dst_n, Index stride_m, Index stride_n, Scalar beta, TileGraph::TileNode* dst)
{
    if(!src1 || !src2 || !dst)
        throw std::invalid_argument("conv2d_inplace");
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph() || src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
        throw std::invalid_argument("conv2d_inplace");
    auto op = std::make_shared<TileConv2dInplaceOp>(src1_m, src1_n, src1_channels, batch, src2_m, src2_n, dilation_m, dilation_n, dst_channels, offset_m, offset_n, alpha, src1, src2, dst_m, dst_n, stride_m, stride_n, beta, dst);
    src1->graph()->add_op(std::move(op));
}
void TileConv2dInplaceOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(s1);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(*this, runtime);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(*this, runtime);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(*this, runtime);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(*this, runtime);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(*this, runtime);
            break;
        case DataType::FP16:
            throw std::runtime_error("FP16 not supported for tile conv2d_inplace in this build");
        case DataType::BF16:
            run<nntile::bf16_t>(*this, runtime);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("conv2d_inplace");
        default:
            throw std::runtime_error("conv2d_inplace");
    }
}
} // namespace nntile::graph::tile_graph
