/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/gemm.hh
 * TileGraph GEMM: C = alpha * op(A) @ op(B) + beta * C
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Tile-level GEMM matching TensorGemmOp semantics.
struct TileGemmOp : TileGraph::OpNode
{
    bool trans_a = false;
    bool trans_b = false;
    Scalar alpha = 1.0;
    Scalar beta = 0.0;
    Index ndim = 1;
    Index batch_ndim = 0;
    TileGraph::TileNode* a = nullptr;
    TileGraph::TileNode* b = nullptr;
    TileGraph::TileNode* c = nullptr;

    TileGemmOp() = default;
    TileGemmOp(
        TileGraph::TileNode* a_,
        TileGraph::TileNode* b_,
        TileGraph::TileNode* c_,
        Scalar alpha_,
        Scalar beta_,
        bool trans_a_,
        bool trans_b_,
        Index ndim_,
        Index batch_ndim_)
        : trans_a(trans_a_)
        , trans_b(trans_b_)
        , alpha(alpha_)
        , beta(beta_)
        , ndim(ndim_)
        , batch_ndim(batch_ndim_)
        , a(a_)
        , b(b_)
        , c(c_)
    {
        inputs_ = {a, b, c};
        outputs_ = {c};
    }

    std::string op_name() const override { return "TILE_GEMM"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileGemmOp>(*this);
    }
};

//! GEMM into existing tile C.
void gemm(
    TileGraph::TileNode* a,
    TileGraph::TileNode* b,
    TileGraph::TileNode* c,
    Scalar alpha,
    Scalar beta,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim);

} // namespace nntile::graph::tile_graph
