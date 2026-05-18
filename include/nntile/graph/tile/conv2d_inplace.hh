/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/conv2d_inplace.hh
 * TileGraph conv2d_inplace: Y = alpha*conv(X,C) + beta*Y
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Conv2D forward: Y = alpha*conv(X,C) + beta*Y
struct TileConv2dInplaceOp : TileGraph::OpNode
{
    Index src1_m = 0, src1_n = 0, src1_channels = 0, batch = 0, src2_m = 0, src2_n = 0, dilation_m = 0, dilation_n = 0, dst_channels = 0, offset_m = 0, offset_n = 0;
    Index dst_m = 0, dst_n = 0, stride_m = 0, stride_n = 0;
    Scalar alpha = 0.0, beta = 0.0;
    TileGraph::TileNode *s1 = nullptr, *s2 = nullptr, *dst = nullptr;
    TileConv2dInplaceOp() = default;
    TileConv2dInplaceOp(Index a1, Index a2, Index a3, Index b, Index w1, Index w2, Index d1, Index d2, Index dc, Index om, Index on, Scalar al, TileGraph::TileNode* t1, TileGraph::TileNode* t2, Index dm, Index dn, Index sm, Index sn, Scalar be, TileGraph::TileNode* d)
        : src1_m(a1)
        , src1_n(a2)
        , src1_channels(a3)
        , batch(b)
        , src2_m(w1)
        , src2_n(w2)
        , dilation_m(d1)
        , dilation_n(d2)
        , dst_channels(dc)
        , offset_m(om)
        , offset_n(on)
        , dst_m(dm)
        , dst_n(dn)
        , stride_m(sm)
        , stride_n(sn)
        , alpha(al)
        , beta(be)
        , s1(t1)
        , s2(t2)
        , dst(d)
    {
        inputs_ = {s1, s2, dst};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_CONV2D_INPLACE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileConv2dInplaceOp>(*this);
    }
};
//! Conv2D forward: Y = alpha*conv(X,C) + beta*Y
void conv2d_inplace(
    Index src1_m, Index src1_n, Index src1_channels, Index batch, Index src2_m, Index src2_n, Index dilation_m, Index dilation_n, Index dst_channels, Index offset_m, Index offset_n, Scalar alpha, TileGraph::TileNode* src1, TileGraph::TileNode* src2, Index dst_m, Index dst_n, Index stride_m, Index stride_n, Scalar beta, TileGraph::TileNode* dst);
} // namespace nntile::graph::tile_graph
