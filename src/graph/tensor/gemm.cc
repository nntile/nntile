/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/gemm.cc
 * TensorGraph GEMM operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/gemm.hh"

#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/constants.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tile/graph_ops.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/tensor/gemm.hh"

namespace nntile::graph::tensor
{

std::vector<Index> gemm_output_shape(
    const std::vector<Index>& a_shape,
    const std::vector<Index>& b_shape,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    Index a_ndim = static_cast<Index>(a_shape.size());
    Index b_ndim = static_cast<Index>(b_shape.size());

    std::vector<Index> output_shape;
    output_shape.reserve(a_ndim + b_ndim - 2 * ndim);

    Index a_batch_start = a_ndim - batch_ndim;
    Index b_batch_start = b_ndim - batch_ndim;

    Index a_m_begin = trans_a ? ndim : 0;
    Index a_m_end = trans_a ? a_batch_start : a_batch_start - ndim;
    Index b_n_begin = trans_b ? 0 : ndim;
    Index b_n_end = trans_b ? b_batch_start - ndim : b_batch_start;

    output_shape.insert(output_shape.end(),
                        a_shape.begin() + a_m_begin,
                        a_shape.begin() + a_m_end);
    output_shape.insert(output_shape.end(),
                        b_shape.begin() + b_n_begin,
                        b_shape.begin() + b_n_end);
    output_shape.insert(output_shape.end(),
                        a_shape.begin() + a_batch_start,
                        a_shape.end());

    return output_shape;
}

namespace
{
constexpr Scalar gemm_new_output_beta = 0.0;

//! Validate A-B, A-C, B-C shapes and merge axes (inline check+merge per dimension).
void validate_gemm_shape_and_merge(
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    TensorGraph::TensorNode* c,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    Index a_ndim = a->ndim();
    Index b_ndim = b->ndim();
    Index c_ndim = c->ndim();
    if(a_ndim < ndim + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: A must have ndim >= ndim + batch_ndim (" +
            std::to_string(a_ndim) + " vs " +
            std::to_string(ndim + batch_ndim) + ")");
    }
    if(b_ndim < ndim + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: B must have ndim >= ndim + batch_ndim (" +
            std::to_string(b_ndim) + " vs " +
            std::to_string(ndim + batch_ndim) + ")");
    }
    Index a_batch_start = a_ndim - batch_ndim;
    Index b_batch_start = b_ndim - batch_ndim;
    Index a_m_begin = trans_a ? ndim : 0;
    Index a_m_end = trans_a ? a_batch_start : a_batch_start - ndim;
    Index a_k_begin = trans_a ? 0 : a_m_end;
    Index b_k_begin = trans_b ? (b_batch_start - ndim) : 0;
    Index b_n_begin = trans_b ? 0 : ndim;
    Index b_n_end = trans_b ? b_batch_start - ndim : b_batch_start;
    Index num_m = a_m_end - a_m_begin;
    Index num_n = b_n_end - b_n_begin;
    Index c_batch_start = num_m + num_n;
    if(c_ndim != c_batch_start + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: C ndim must equal num_m + num_n + batch_ndim (" +
            std::to_string(c_ndim) + " vs " +
            std::to_string(c_batch_start + batch_ndim) + ")");
    }
    // A-B: contracted (K) dimensions
    for(Index i = 0; i < ndim; ++i)
    {
        if(a->shape()[a_k_begin + i] != b->shape()[b_k_begin + i])
        {
            throw std::invalid_argument(
                "gemm: contracted dimension " + std::to_string(i) +
                " must match (A: " + std::to_string(a->shape()[a_k_begin + i]) +
                " vs B: " + std::to_string(b->shape()[b_k_begin + i]) + ")");
        }
        merge_axis(a->mutable_axes()[a_k_begin + i],
                   b->mutable_axes()[b_k_begin + i]);
    }
    // A-C: M dimensions
    for(Index i = 0; i < num_m; ++i)
    {
        if(a->shape()[a_m_begin + i] != c->shape()[i])
        {
            throw std::invalid_argument(
                "gemm: M dimension " + std::to_string(i) +
                " must match (A: " + std::to_string(a->shape()[a_m_begin + i]) +
                " vs C: " + std::to_string(c->shape()[i]) + ")");
        }
        merge_axis(a->mutable_axes()[a_m_begin + i], c->mutable_axes()[i]);
    }
    // B-C: N dimensions
    for(Index i = 0; i < num_n; ++i)
    {
        if(b->shape()[b_n_begin + i] != c->shape()[num_m + i])
        {
            throw std::invalid_argument(
                "gemm: N dimension " + std::to_string(i) +
                " must match (B: " + std::to_string(b->shape()[b_n_begin + i]) +
                " vs C: " + std::to_string(c->shape()[num_m + i]) + ")");
        }
        merge_axis(b->mutable_axes()[b_n_begin + i],
                   c->mutable_axes()[num_m + i]);
    }
    // A-B-C: batch dimensions
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(a->shape()[a_batch_start + i] != c->shape()[c_batch_start + i])
        {
            throw std::invalid_argument(
                "gemm: batch dimension " + std::to_string(i) +
                " must match (A: " + std::to_string(a->shape()[a_batch_start + i]) +
                " vs C: " + std::to_string(c->shape()[c_batch_start + i]) + ")");
        }
        merge_axis(a->mutable_axes()[a_batch_start + i],
                   c->mutable_axes()[c_batch_start + i]);
        if(b->shape()[b_batch_start + i] != c->shape()[c_batch_start + i])
        {
            throw std::invalid_argument(
                "gemm: batch dimension " + std::to_string(i) +
                " must match (B: " + std::to_string(b->shape()[b_batch_start + i]) +
                " vs C: " + std::to_string(c->shape()[c_batch_start + i]) + ")");
        }
        merge_axis(b->mutable_axes()[b_batch_start + i],
                   c->mutable_axes()[c_batch_start + i]);
    }
}

template<typename T>
void run_gemm(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    bool trans_a, bool trans_b,
    Index ndim, Index batch_ndim,
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    TensorGraph::TensorNode* c)
{
    auto& a_t = runtime.get_tensor<T>(a);
    auto& b_t = runtime.get_tensor<T>(b);
    auto& c_t = runtime.get_tensor<T>(c);

    const auto trans_a_op = trans_a ?
        nntile::TransOp(nntile::TransOp::Trans) :
        nntile::TransOp(nntile::TransOp::NoTrans);
    const auto trans_b_op = trans_b ?
        nntile::TransOp(nntile::TransOp::Trans) :
        nntile::TransOp(nntile::TransOp::NoTrans);

    nntile::tensor::gemm<T>(
        alpha, trans_a_op, a_t, trans_b_op, b_t,
        beta, c_t, ndim, batch_ndim, 0);
}

} // namespace

TensorGraph::TensorNode* gemm(
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    const std::string& output_name,
    Scalar alpha,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    if(a == nullptr || b == nullptr)
    {
        throw std::invalid_argument("gemm: input tensors must be non-null");
    }
    if(a->graph() != b->graph())
    {
        throw std::invalid_argument(
            "gemm: input tensors must belong to the same graph");
    }
    if(a->dtype() != b->dtype())
    {
        throw std::invalid_argument(
            "gemm: input tensors must have the same dtype");
    }
    if(a->ndim() < ndim + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: A must have ndim >= ndim + batch_ndim (" +
            std::to_string(a->ndim()) + " vs " +
            std::to_string(ndim + batch_ndim) + ")");
    }
    if(b->ndim() < ndim + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: B must have ndim >= ndim + batch_ndim (" +
            std::to_string(b->ndim()) + " vs " +
            std::to_string(ndim + batch_ndim) + ")");
    }

    std::vector<Index> output_shape = gemm_output_shape(
        a->shape(), b->shape(), trans_a, trans_b, ndim, batch_ndim);

    TensorGraph::TensorNode* output = a->graph()->data(
        std::move(output_shape),
        output_name,
        a->dtype());

    gemm(a, b, output, alpha, gemm_new_output_beta, trans_a, trans_b, ndim,
         batch_ndim);

    return output;
}

void gemm(
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    TensorGraph::TensorNode* c,
    Scalar alpha,
    Scalar beta,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    if(a == nullptr || b == nullptr || c == nullptr)
    {
        throw std::invalid_argument("gemm: input tensors must be non-null");
    }
    if(a->graph() != b->graph() || a->graph() != c->graph())
    {
        throw std::invalid_argument(
            "gemm: input tensors must belong to the same graph");
    }
    if(a->dtype() != b->dtype() || a->dtype() != c->dtype())
    {
        throw std::invalid_argument(
            "gemm: input tensors must have the same dtype");
    }

    validate_gemm_shape_and_merge(a, b, c, trans_a, trans_b, ndim, batch_ndim);

    auto op = std::make_shared<TensorGemmOp>(
        a, b, c, alpha, beta, trans_a, trans_b, ndim, batch_ndim);

    a->graph()->add_op(op);
}

void TensorGemmOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(a);

    switch(dtype)
    {
        case DataType::FP32:
            run_gemm<nntile::fp32_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::FP32_FAST_TF32:
            run_gemm<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::FP32_FAST_FP16:
            run_gemm<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::FP32_FAST_BF16:
            run_gemm<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::FP64:
            run_gemm<nntile::fp64_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::FP16:
            run_gemm<nntile::fp16_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::BF16:
            run_gemm<nntile::bf16_t>(
                runtime, alpha, beta, trans_a, trans_b, ndim, batch_ndim,
                a, b, c);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gemm operation");
        default:
            throw std::runtime_error("Unsupported data type for gemm");
    }
}

namespace
{

struct GemmAxisRoles
{
    Index a_ndim = 0;
    Index b_ndim = 0;
    Index c_ndim = 0;
    Index a_batch_start = 0;
    Index b_batch_start = 0;
    Index a_m_begin = 0;
    Index a_m_end = 0;
    Index a_k_begin = 0;
    Index b_k_begin = 0;
    Index b_n_begin = 0;
    Index b_n_end = 0;
    Index num_m = 0;
    Index num_n = 0;
    Index c_batch_start = 0;
    bool trans_a = false;
    bool trans_b = false;
    Index ndim = 1;
    Index batch_ndim = 0;

    GemmAxisRoles(
        const TensorGraph::TensorNode* a,
        const TensorGraph::TensorNode* b,
        const TensorGraph::TensorNode* c,
        bool ta,
        bool tb,
        Index nd,
        Index bd)
        : trans_a(ta)
        , trans_b(tb)
        , ndim(nd)
        , batch_ndim(bd)
    {
        a_ndim = a->ndim();
        b_ndim = b->ndim();
        c_ndim = c->ndim();
        a_batch_start = a_ndim - batch_ndim;
        b_batch_start = b_ndim - batch_ndim;
        a_m_begin = trans_a ? ndim : 0;
        a_m_end = trans_a ? a_batch_start : a_batch_start - ndim;
        a_k_begin = trans_a ? 0 : a_m_end;
        b_k_begin = trans_b ? (b_batch_start - ndim) : 0;
        b_n_begin = trans_b ? 0 : ndim;
        b_n_end = trans_b ? b_batch_start - ndim : b_batch_start;
        num_m = a_m_end - a_m_begin;
        num_n = b_n_end - b_n_begin;
        c_batch_start = num_m + num_n;
        if(c_ndim != c_batch_start + batch_ndim)
        {
            throw std::invalid_argument(
                "GEMM lowering: C ndim mismatch for GEMM layout");
        }
    }

    Index a_axis_m(Index i) const { return a_m_begin + i; }
    Index a_axis_k(Index i) const { return a_k_begin + i; }
    Index a_axis_batch(Index j) const { return a_batch_start + j; }

    Index b_axis_k(Index i) const { return b_k_begin + i; }
    Index b_axis_n(Index i) const { return b_n_begin + i; }
    Index b_axis_batch(Index j) const { return b_batch_start + j; }

    Index c_axis_m(Index i) const { return i; }
    Index c_axis_n(Index i) const { return num_m + i; }
    Index c_axis_batch(Index j) const { return c_batch_start + j; }
};

void tile_bbox(
    const TensorAxisLayout& lay,
    Index linear,
    std::vector<Index>& lo,
    std::vector<Index>& hi)
{
    std::vector<Index> gc;
    lay.grid_coord_from_linear(linear, gc);
    const Index nd = static_cast<Index>(lay.tensor_shape().size());
    lo.resize(static_cast<size_t>(nd));
    hi.resize(static_cast<size_t>(nd));
    for(Index d = 0; d < nd; ++d)
    {
        lay.tile_axis_global_range(gc, d, lo[static_cast<size_t>(d)],
            hi[static_cast<size_t>(d)]);
    }
}

Index find_exact_tile(
    const TensorAxisLayout& lay,
    const std::vector<Index>& req_lo,
    const std::vector<Index>& req_hi,
    const std::string& tensor_name)
{
    std::vector<Index> lo, hi;
    for(Index lin = 0; lin < lay.grid_volume(); ++lin)
    {
        tile_bbox(lay, lin, lo, hi);
        bool ok = true;
        for(size_t d = 0; d < lo.size(); ++d)
        {
            if(lo[d] != req_lo[d] || hi[d] != req_hi[d])
            {
                ok = false;
                break;
            }
        }
        if(ok)
        {
            return lin;
        }
    }
    std::ostringstream oss;
    oss << "GEMM lowering: no tile of '" << tensor_name
        << "' matches required GEMM slice (unsupported tiling / alignment)";
    throw std::runtime_error(oss.str());
}

void set_full_range(
    const TensorGraph::TensorNode* t,
    std::vector<Index>& lo,
    std::vector<Index>& hi)
{
    const Index nd = t->ndim();
    lo.resize(static_cast<size_t>(nd));
    hi.resize(static_cast<size_t>(nd));
    for(Index d = 0; d < nd; ++d)
    {
        lo[static_cast<size_t>(d)] = 0;
        hi[static_cast<size_t>(d)] = t->shape()[static_cast<size_t>(d)] - 1;
    }
}

void narrow_axis(
    std::vector<Index>& lo,
    std::vector<Index>& hi,
    Index axis,
    Index new_lo,
    Index new_hi)
{
    lo[static_cast<size_t>(axis)] = new_lo;
    hi[static_cast<size_t>(axis)] = new_hi;
}

Index k_index_volume(const TensorAxisLayout& La, const GemmAxisRoles& geom)
{
    Index v = 1;
    for(Index i = 0; i < geom.ndim; ++i)
    {
        v *= La.grid_shape()[static_cast<size_t>(geom.a_axis_k(i))];
    }
    return v;
}

void decode_k_index(
    const TensorAxisLayout& La,
    const GemmAxisRoles& geom,
    Index kk,
    std::vector<Index>& k_coord)
{
    k_coord.assign(static_cast<size_t>(geom.ndim), 0);
    Index rem = kk;
    for(Index i = 0; i < geom.ndim; ++i)
    {
        Index stride = 1;
        for(Index j = i + 1; j < geom.ndim; ++j)
        {
            stride *= La.grid_shape()[static_cast<size_t>(geom.a_axis_k(j))];
        }
        k_coord[static_cast<size_t>(i)] = rem / stride;
        rem %= stride;
    }
}

} // namespace

void TensorGemmOp::lower_to_tile(const LoweringContext& ctx) const
{
    const TensorAxisLayout* La = ctx.tiling.find(a);
    const TensorAxisLayout* Lb = ctx.tiling.find(b);
    const TensorAxisLayout* Lc = ctx.tiling.find(c);
    if(La == nullptr || Lb == nullptr || Lc == nullptr)
    {
        throw std::runtime_error("GEMM lowering: missing TensorAxisLayout");
    }

    const std::vector<TileGraph::TileNode*>& va =
        tile_lower::tiles_of(ctx.tile_map, a);
    const std::vector<TileGraph::TileNode*>& vb =
        tile_lower::tiles_of(ctx.tile_map, b);
    const std::vector<TileGraph::TileNode*>& vc =
        tile_lower::tiles_of(ctx.tile_map, c);

    if(!va.empty() && va[0]->graph() != &ctx.out)
    {
        throw std::runtime_error(
            "GEMM lowering: tile map tensors are not from ctx.out");
    }

    GemmAxisRoles geom(a, b, c, trans_a, trans_b, ndim, batch_ndim);

    const Index k_vol = k_index_volume(*La, geom);

    std::vector<Index> c_lo, c_hi;
    std::vector<Index> req_a_lo, req_a_hi, req_b_lo, req_b_hi;
    std::vector<Index> k_coord;

    for(Index lin_c = 0; lin_c < Lc->grid_volume(); ++lin_c)
    {
        tile_bbox(*Lc, lin_c, c_lo, c_hi);

        for(Index kk = 0; kk < k_vol; ++kk)
        {
            decode_k_index(*La, geom, kk, k_coord);

            set_full_range(a, req_a_lo, req_a_hi);
            for(Index i = 0; i < geom.num_m; ++i)
            {
                const Index ax = geom.a_axis_m(i);
                const Index cx = geom.c_axis_m(i);
                narrow_axis(
                    req_a_lo,
                    req_a_hi,
                    ax,
                    c_lo[static_cast<size_t>(cx)],
                    c_hi[static_cast<size_t>(cx)]);
            }
            for(Index i = 0; i < geom.ndim; ++i)
            {
                const Index ax = geom.a_axis_k(i);
                std::vector<Index> gtmp(La->tensor_shape().size(), 0);
                gtmp[static_cast<size_t>(ax)] = k_coord[static_cast<size_t>(i)];
                Index kl = 0, kh = 0;
                La->tile_axis_global_range(
                    gtmp, ax, kl, kh);
                narrow_axis(req_a_lo, req_a_hi, ax, kl, kh);
            }
            for(Index j = 0; j < geom.batch_ndim; ++j)
            {
                const Index ax = geom.a_axis_batch(j);
                const Index cx = geom.c_axis_batch(j);
                narrow_axis(
                    req_a_lo,
                    req_a_hi,
                    ax,
                    c_lo[static_cast<size_t>(cx)],
                    c_hi[static_cast<size_t>(cx)]);
            }

            set_full_range(b, req_b_lo, req_b_hi);
            for(Index i = 0; i < geom.ndim; ++i)
            {
                const Index bx = geom.b_axis_k(i);
                std::vector<Index> gtmp(Lb->tensor_shape().size(), 0);
                gtmp[static_cast<size_t>(bx)] = k_coord[static_cast<size_t>(i)];
                Index kl = 0, kh = 0;
                Lb->tile_axis_global_range(
                    gtmp, bx, kl, kh);
                narrow_axis(req_b_lo, req_b_hi, bx, kl, kh);
            }
            for(Index i = 0; i < geom.num_n; ++i)
            {
                const Index bx = geom.b_axis_n(i);
                const Index cx = geom.c_axis_n(i);
                narrow_axis(
                    req_b_lo,
                    req_b_hi,
                    bx,
                    c_lo[static_cast<size_t>(cx)],
                    c_hi[static_cast<size_t>(cx)]);
            }
            for(Index j = 0; j < geom.batch_ndim; ++j)
            {
                const Index bx = geom.b_axis_batch(j);
                const Index cx = geom.c_axis_batch(j);
                narrow_axis(
                    req_b_lo,
                    req_b_hi,
                    bx,
                    c_lo[static_cast<size_t>(cx)],
                    c_hi[static_cast<size_t>(cx)]);
            }

            const Index lin_a = find_exact_tile(*La, req_a_lo, req_a_hi, a->name());
            const Index lin_b = find_exact_tile(*Lb, req_b_lo, req_b_hi, b->name());

            TileGraph::TileNode* ta = va[static_cast<size_t>(lin_a)];
            TileGraph::TileNode* tb = vb[static_cast<size_t>(lin_b)];
            TileGraph::TileNode* tc = vc[static_cast<size_t>(lin_c)];

            const bool first_k = (kk == 0);
            const Scalar beta_use = (first_k ? beta : Scalar(1.0));

            tile_graph::gemm(
                ta,
                tb,
                tc,
                alpha,
                beta_use,
                trans_a,
                trans_b,
                ndim,
                batch_ndim);
        }
    }
}

} // namespace nntile::graph::tensor
