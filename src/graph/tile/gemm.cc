/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/gemm.cc
 * TileGraph GEMM implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/gemm.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/constants.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/gemm.hh>

namespace nntile::graph::tile_graph
{

namespace
{

template<typename T>
void run_gemm(
    TileGraph::Runtime& runtime,
    Scalar alpha,
    Scalar beta,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim,
    TileGraph::TileNode* a,
    TileGraph::TileNode* b,
    TileGraph::TileNode* c)
{
    auto& a_t = runtime.get_tile<T>(a);
    auto& b_t = runtime.get_tile<T>(b);
    auto& c_t = runtime.get_tile<T>(c);

    const auto trans_a_op = trans_a ? nntile::TransOp(nntile::TransOp::Trans)
                                    : nntile::TransOp(nntile::TransOp::NoTrans);
    const auto trans_b_op = trans_b ? nntile::TransOp(nntile::TransOp::Trans)
                                    : nntile::TransOp(nntile::TransOp::NoTrans);

    nntile::tile::gemm<T>(
        alpha, trans_a_op, a_t, trans_b_op, b_t, beta, c_t, ndim, batch_ndim,
        0);
}

} // namespace

void gemm(
    TileGraph::TileNode* a,
    TileGraph::TileNode* b,
    TileGraph::TileNode* c,
    Scalar alpha,
    Scalar beta,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    if(a == nullptr || b == nullptr || c == nullptr)
    {
        throw std::invalid_argument("tile gemm: a, b, c must be non-null");
    }
    if(a->graph() != b->graph() || a->graph() != c->graph())
    {
        throw std::invalid_argument(
            "tile gemm: operands must belong to the same graph");
    }
    if(a->dtype() != b->dtype() || a->dtype() != c->dtype())
    {
        throw std::invalid_argument("tile gemm: dtype mismatch");
    }

    auto op = std::make_shared<TileGemmOp>(
        a, b, c, alpha, beta, trans_a, trans_b, ndim, batch_ndim);
    a->graph()->add_op(op);
}

void TileGemmOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(a);

    switch(dtype)
    {
        case DataType::FP32:
            run_gemm<nntile::fp32_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim, a,
                b, c);
            break;
        case DataType::FP32_FAST_TF32:
            run_gemm<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim, a,
                b, c);
            break;
        case DataType::FP32_FAST_FP16:
            run_gemm<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim, a,
                b, c);
            break;
        case DataType::FP32_FAST_BF16:
            run_gemm<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim, a,
                b, c);
            break;
        case DataType::FP64:
            run_gemm<nntile::fp64_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim, a,
                b, c);
            break;
        case DataType::FP16:
            run_gemm<nntile::fp16_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim, a,
                b, c);
            break;
        case DataType::BF16:
            run_gemm<nntile::bf16_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim, a,
                b, c);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile GEMM");
        default:
            throw std::runtime_error("Unsupported data type for tile GEMM");
    }
}

} // namespace nntile::graph::tile_graph
