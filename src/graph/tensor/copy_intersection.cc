/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/copy_intersection.cc
 * TensorGraph copy_intersection operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/copy_intersection.hh"

#include <algorithm>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/copy.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/copy_intersection.hh"

namespace nntile::graph::tensor
{

namespace
{

//! Per-axis global intersection of src/dst in tensor coordinates (same rules
//! as nntile::tensor::copy_intersection). Returns false if no overlap.
bool compute_global_intersection(
    const std::vector<Index>& src_shape,
    const std::vector<Index>& dst_shape,
    const std::vector<Index>& op_src_offset,
    const std::vector<Index>& op_dst_offset,
    std::vector<Index>& g_src_start,
    std::vector<Index>& g_dst_start,
    std::vector<Index>& copy_shape)
{
    if(src_shape.empty())
    {
        g_src_start.clear();
        g_dst_start.clear();
        copy_shape.clear();
        return true;
    }
    const Index ndim = static_cast<Index>(src_shape.size());
    g_src_start.resize(static_cast<size_t>(ndim));
    g_dst_start.resize(static_cast<size_t>(ndim));
    copy_shape.resize(static_cast<size_t>(ndim));
    for(Index i = 0; i < ndim; ++i)
    {
        if(op_src_offset[static_cast<size_t>(i)] + src_shape[static_cast<size_t>(i)]
                <= op_dst_offset[static_cast<size_t>(i)]
            || op_dst_offset[static_cast<size_t>(i)]
                    + dst_shape[static_cast<size_t>(i)]
                <= op_src_offset[static_cast<size_t>(i)])
        {
            return false;
        }
        if(op_src_offset[static_cast<size_t>(i)]
            < op_dst_offset[static_cast<size_t>(i)])
        {
            g_src_start[static_cast<size_t>(i)] =
                op_dst_offset[static_cast<size_t>(i)]
                - op_src_offset[static_cast<size_t>(i)];
            g_dst_start[static_cast<size_t>(i)] = 0;
            copy_shape[static_cast<size_t>(i)] = std::min(
                src_shape[static_cast<size_t>(i)]
                    - g_src_start[static_cast<size_t>(i)],
                dst_shape[static_cast<size_t>(i)]);
        }
        else
        {
            g_src_start[static_cast<size_t>(i)] = 0;
            g_dst_start[static_cast<size_t>(i)] =
                op_src_offset[static_cast<size_t>(i)]
                - op_dst_offset[static_cast<size_t>(i)];
            copy_shape[static_cast<size_t>(i)] = std::min(
                dst_shape[static_cast<size_t>(i)]
                    - g_dst_start[static_cast<size_t>(i)],
                src_shape[static_cast<size_t>(i)]);
        }
        if(copy_shape[static_cast<size_t>(i)] == 0)
        {
            return false;
        }
    }
    return true;
}

bool same_shape_same_offset_fast_path_tiling(
    const TensorGraph::TensorNode* a, const TensorGraph::TensorNode* b)
{
    if(a->shape() != b->shape())
    {
        return false;
    }
    if(a->ndim() != b->ndim())
    {
        return false;
    }
    for(Index d = 0; d < a->ndim(); ++d)
    {
        if(a->axis(static_cast<int>(d))->tile_sizes
            != b->axis(static_cast<int>(d))->tile_sizes)
        {
            return false;
        }
    }
    return true;
}

} // namespace

void TensorCopyIntersectionOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& tsrc = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tdst = tile_lower::tiles_of(ctx.tile_map, dst);

    // Copy_intersection "easy" case: aligned offsets, same shape, same
    // elementwise layout -> per-tile copy (nntile::tensor::copy_intersection
    // fast path, src/tensor/copy_intersection.cc).
    if(src_offset == dst_offset && same_shape_same_offset_fast_path_tiling(
                                    src, dst)
        && tsrc.size() == tdst.size())
    {
        for(size_t i = 0; i < tsrc.size(); ++i)
        {
            tile_graph::copy(tsrc[i], tdst[i]);
        }
        return;
    }

    const TensorAxisLayout* lay_s = ctx.tiling.find(src);
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_s == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile COPY_INTERSECTION: missing tiling for src and/or "
            "dst");
    }
    const std::vector<Index>& s_shape = src->shape();
    const std::vector<Index>& d_shape = dst->shape();
    std::vector<Index> g_src, g_dst, cshape;
    if(!compute_global_intersection(
            s_shape, d_shape, src_offset, dst_offset, g_src, g_dst, cshape))
    {
        return;
    }
    const Index ndim = src->ndim();
    if(ndim == 0)
    {
        if(tsrc.size() == 1 && tdst.size() == 1)
        {
            tile_graph::copy(tsrc[0], tdst[0]);
        }
        return;
    }

    const std::string scratch_name = std::string("__cpis_scr_")
        + std::to_string(reinterpret_cast<std::uintptr_t>(this));
    TileGraph::TileNode* scratch = ctx.out.data(
        std::vector<Index>{2 * ndim}, scratch_name, DataType::INT64);

    // Nested dst-tile / src-tile iteration aligned with
    // nntile::tensor::copy_intersection (src/tensor/copy_intersection.cc),
    // using TensorAxisLayout segment origins instead of uniform basetiles.
    std::vector<Index> dst_tile_index_begin(static_cast<size_t>(ndim));
    std::vector<Index> dst_tile_index_end(static_cast<size_t>(ndim));
    Index dst_ntiles = 1;
    for(Index j = 0; j < ndim; ++j)
    {
        const size_t jz = static_cast<size_t>(j);
        dst_tile_index_begin[jz] =
            lay_d->tile_index_containing(j, g_dst[jz]);
        dst_tile_index_end[jz] = lay_d->tile_index_containing(
                                    j, g_dst[jz] + cshape[jz] - 1)
            + 1;
        dst_ntiles *= dst_tile_index_end[jz] - dst_tile_index_begin[jz];
    }
    std::vector<Index> dst_tile_index = dst_tile_index_begin;
    std::vector<Index> src_tile_index_begin(static_cast<size_t>(ndim));
    std::vector<Index> src_tile_index_end(static_cast<size_t>(ndim));
    std::vector<Index> src_tile_index(static_cast<size_t>(ndim));
    std::vector<Index> src_corner(static_cast<size_t>(ndim));
    std::vector<Index> dst_corner(static_cast<size_t>(ndim));
    for(Index i = 0; i < dst_ntiles; ++i)
    {
        Index src_ntiles = 1;
        for(Index j = 0; j < ndim; ++j)
        {
            const size_t jz = static_cast<size_t>(j);
            Index d_lo = 0;
            Index d_hi = 0;
            lay_d->tile_axis_global_range(dst_tile_index, j, d_lo, d_hi);
            if(dst_tile_index[jz] == dst_tile_index_begin[jz])
            {
                src_tile_index_begin[jz] =
                    lay_s->tile_index_containing(j, g_src[jz]);
            }
            else
            {
                src_tile_index_begin[jz] = lay_s->tile_index_containing(
                    j, d_lo - g_dst[jz] + g_src[jz]);
            }
            if(dst_tile_index[jz] + 1 == dst_tile_index_end[jz])
            {
                src_tile_index_end[jz] =
                    lay_s->tile_index_containing(
                        j, g_src[jz] + cshape[jz] - 1)
                    + 1;
            }
            else
            {
                src_tile_index_end[jz] = lay_s->tile_index_containing(
                    j, d_hi - g_dst[jz] + g_src[jz])
                    + 1;
            }
            src_ntiles *= src_tile_index_end[jz] - src_tile_index_begin[jz];
        }
        src_tile_index = src_tile_index_begin;
        for(Index j = 0; j < src_ntiles; ++j)
        {
            const Index lin_s = lay_s->grid_linear(src_tile_index);
            const Index lin_d = lay_d->grid_linear(dst_tile_index);
            for(Index k = 0; k < ndim; ++k)
            {
                const size_t kz = static_cast<size_t>(k);
                Index s_lo = 0;
                Index s_hi_incl = 0;
                lay_s->tile_axis_global_range(
                    src_tile_index, k, s_lo, s_hi_incl);
                (void)s_hi_incl;
                src_corner[kz] = s_lo;
                Index d_lo2 = 0;
                Index d_hi_incl2 = 0;
                lay_d->tile_axis_global_range(
                    dst_tile_index, k, d_lo2, d_hi_incl2);
                (void)d_hi_incl2;
                dst_corner[kz] = d_lo2 - g_dst[kz] + g_src[kz];
            }
            tile_graph::copy_intersection(tsrc[static_cast<size_t>(lin_s)],
                src_corner, tdst[static_cast<size_t>(lin_d)], dst_corner,
                scratch);
            if(j + 1 < src_ntiles)
            {
                ++src_tile_index[0];
                Index k = 0;
                while(src_tile_index[static_cast<size_t>(k)]
                    == src_tile_index_end[static_cast<size_t>(k)])
                {
                    src_tile_index[static_cast<size_t>(k)] =
                        src_tile_index_begin[static_cast<size_t>(k)];
                    ++k;
                    ++src_tile_index[static_cast<size_t>(k)];
                }
            }
        }
        if(i == dst_ntiles - 1)
        {
            break;
        }
        ++dst_tile_index[0];
        Index k = 0;
        while(dst_tile_index[static_cast<size_t>(k)]
            == dst_tile_index_end[static_cast<size_t>(k)])
        {
            dst_tile_index[static_cast<size_t>(k)] =
                dst_tile_index_begin[static_cast<size_t>(k)];
            ++k;
            ++dst_tile_index[static_cast<size_t>(k)];
        }
    }
}

void copy_intersection(TensorGraph::TensorNode* src,
                       const std::vector<Index>& src_offset,
                       TensorGraph::TensorNode* dst,
                       const std::vector<Index>& dst_offset)
{
    if(src == nullptr || dst == nullptr)
        throw std::invalid_argument(
            "copy_intersection: tensors must be non-null");
    if(src->graph() != dst->graph())
        throw std::invalid_argument(
            "copy_intersection: tensors must belong to same graph");
    if(src->dtype() != dst->dtype())
        throw std::invalid_argument(
            "copy_intersection: tensors must have same dtype");
    if(src_offset.size() != src->ndim() || dst_offset.size() != dst->ndim())
        throw std::invalid_argument(
            "copy_intersection: offset sizes must match tensor ndim");
    auto op = std::make_shared<TensorCopyIntersectionOp>(
        src, src_offset, dst, dst_offset);
    dst->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
