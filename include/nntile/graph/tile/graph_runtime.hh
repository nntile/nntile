/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/graph_runtime.hh
 * TileGraphExecutor - compile/execute a TileGraph (alias ``TileGraph::Runtime``).
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Third-party headers
#include <starpu.h>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile/graph_decl.hh>
#include <nntile/graph/tensor/graph_data_node.hh>
#include <nntile/graph/tensor/tensor_graph_tiling.hh>
#include <nntile/graph/tile/graph_data_node.hh>
#include <nntile/graph/nn/graph_decl.hh>
#include <nntile/tile/tile.hh>

namespace nntile::graph
{

//! StarPU-backed executor for a TileGraph (IR is separate).
class TileGraphExecutor
{
public:
    using TileNode = TileGraph::TileNode;
    using OpNode = TileGraph::OpNode;

    explicit TileGraphExecutor(const TileGraph& graph);

    void compile();

    //! Run ops [op_begin, op_end) in the post-dead-code elimination order.
    void execute_range(size_t op_begin, size_t op_end);

    size_t execution_op_count() const { return execution_order_.size(); }

    //! Bind host data to a logical tensor or scatter to its tiles.
    template<typename T>
    void bind_data(TensorGraph::TensorNode const* tensor, const T* data,
                   size_t count);

    template<typename T>
    void bind_data(TensorGraph::TensorNode const* tensor,
                   const std::vector<T>& data);

    //! Bind via ``NNGraph::TensorNode`` (same as ``tensor->data()``).
    template<typename T>
    void bind_data(NNGraph::TensorNode const* tensor, const T* data,
                   size_t count);

    template<typename T>
    void bind_data(NNGraph::TensorNode const* tensor,
                   const std::vector<T>& data);

    //! Bind host data to a standalone tile (no tensor descriptor).
    template<typename T>
    void bind_data(TileNode const* tile, const T* data, size_t count);

    template<typename T>
    void bind_data(TileNode const* tile, const std::vector<T>& data);

    void execute();

    void wait();

    //! Read a logical tensor or one tile buffer marked as output.
    template<typename T>
    std::vector<T> get_output(TensorGraph::TensorNode const* tensor);

    template<typename T>
    std::vector<T> get_output(NNGraph::TensorNode const* tensor);

    template<typename T>
    std::vector<T> get_output(TileNode const* tile);

    template<typename T>
    nntile::tile::Tile<T>& get_tile(const TileNode* node);

    DataType get_dtype(TensorGraph::TensorNode const* tensor) const;

    DataType get_dtype(NNGraph::TensorNode const* tensor) const;

    DataType get_dtype(const TileNode* node) const
    {
        return node->dtype();
    }

    bool is_compiled() const { return compiled_; }

private:
    void allocate_missing_tiles();
    void eliminate_dead_ops();

    template<typename T>
    nntile::tile::Tile<T>& get_data(const std::string& runtime_key);

    template<typename T, typename NntileT, typename CastT>
    void bind_data_impl(const std::string& runtime_key, const T* data,
                        size_t count);
    template<typename T, typename NntileT, typename CastT>
    void get_output_impl(const std::string& runtime_key,
                         std::vector<T>& result);

    const TileGraph& graph_;
    std::map<const TileNode*, std::shared_ptr<void>> tile_map_;
    std::map<std::string, std::shared_ptr<void>> runtime_data_;
    std::map<std::string, DataType> data_dtypes_;
    std::vector<std::shared_ptr<OpNode>> execution_order_;
    std::set<std::string> data_is_input_;
    std::set<std::string> data_is_output_;
    bool compiled_ = false;
};

} // namespace nntile::graph

#include <nntile/graph/nn/graph_data_node.hh>

namespace nntile::graph
{

// ---------------------------------------------------------------------------
// TileGraphExecutor template implementation
// ---------------------------------------------------------------------------

template<typename T>
nntile::tile::Tile<T>& TileGraphExecutor::get_data(
    const std::string& runtime_key)
{
    auto it = runtime_data_.find(runtime_key);
    if(it == runtime_data_.end())
    {
        throw std::runtime_error(
            "TileGraphExecutor: data not found: " + runtime_key);
    }
    return *static_cast<nntile::tile::Tile<T>*>(it->second.get());
}

namespace tile_graph_bind_detail
{

inline bool tensor_desc_has_input_tile(
    TileGraph::TensorDescriptor const& d)
{
    for(TileGraph::TileNode* t : d.tiles)
    {
        if(t != nullptr && t->is_input())
        {
            return true;
        }
    }
    return false;
}

inline bool tensor_desc_has_output_tile(
    TileGraph::TensorDescriptor const& d)
{
    for(TileGraph::TileNode* t : d.tiles)
    {
        if(t != nullptr && t->is_output())
        {
            return true;
        }
    }
    return false;
}

inline bool use_logical_layout(
    TileGraph::TensorDescriptor const* desc,
    TensorGraph::TensorNode const* tensor)
{
    if(desc == nullptr || tensor == nullptr
        || desc->source_node != tensor)
    {
        return false;
    }
    return desc->tiles.size() > static_cast<size_t>(1)
        || (desc->tiles.size() == static_cast<size_t>(1)
            && desc->tiles[0]->name() != tensor->name());
}

} // namespace tile_graph_bind_detail

namespace tile_detail
{

template<typename T>
struct dtype_for
{
    static_assert(sizeof(T) == 0,
                  "Unsupported tile type for get_tile; use fp32_t, fp64_t, "
                  "fp16_t, bf16_t, int64_t, bool_t, or fp32_fast_* variants");
};

template<> struct dtype_for<nntile::fp32_t>
{
    static constexpr DataType value = DataType::FP32;
};
template<> struct dtype_for<nntile::fp32_fast_tf32_t>
{
    static constexpr DataType value = DataType::FP32_FAST_TF32;
};
template<> struct dtype_for<nntile::fp32_fast_fp16_t>
{
    static constexpr DataType value = DataType::FP32_FAST_FP16;
};
template<> struct dtype_for<nntile::fp32_fast_bf16_t>
{
    static constexpr DataType value = DataType::FP32_FAST_BF16;
};
template<> struct dtype_for<nntile::fp64_t>
{
    static constexpr DataType value = DataType::FP64;
};
template<> struct dtype_for<nntile::fp16_t>
{
    static constexpr DataType value = DataType::FP16;
};
template<> struct dtype_for<nntile::bf16_t>
{
    static constexpr DataType value = DataType::BF16;
};
template<> struct dtype_for<nntile::int64_t>
{
    static constexpr DataType value = DataType::INT64;
};
template<> struct dtype_for<nntile::bool_t>
{
    static constexpr DataType value = DataType::BOOL;
};

} // namespace tile_detail

namespace tile_graph_layout_io
{

//! Decode a flat offset into tile-local coordinates matching
//! nntile::tile::TileTraits / tile storage (Fortran order: dim 0 stride 1).
inline void fortran_tile_linear_to_index(
    Index linear_offset,
    const std::vector<Index>& shape,
    std::vector<Index>& index)
{
    const size_t ndim = shape.size();
    index.resize(ndim);
    if(ndim == 0)
    {
        return;
    }
    std::vector<Index> stride(ndim);
    stride[0] = 1;
    for(size_t i = 1; i < ndim; ++i)
    {
        stride[i] = stride[i - 1] * shape[i - 1];
    }
    Index rem = linear_offset;
    for(size_t i = ndim - 1; i >= 1; --i)
    {
        const Index div = rem / stride[i];
        rem -= div * stride[i];
        index[i] = div;
    }
    index[0] = rem;
}

//! Dense offset matching logical bind_data / get_output flat layout
//! order (same as nntile tile/tensor Fortran linearization).
inline Index fortran_dense_linear_index(
    const std::vector<Index>& shape,
    const std::vector<Index>& global_coord)
{
    if(shape.size() != global_coord.size())
    {
        throw std::invalid_argument(
            "fortran_dense_linear_index: shape/coord size mismatch");
    }
    Index idx = 0;
    Index stride = 1;
    for(size_t d = 0; d < shape.size(); ++d)
    {
        const Index g = global_coord[d];
        if(g < 0 || g >= shape[d])
        {
            throw std::out_of_range(
                "fortran_dense_linear_index: global coord OOB");
        }
        idx += g * stride;
        stride *= shape[d];
    }
    return idx;
}

template<typename T, typename NntileT, typename CastT>
void scatter_logical_tensor(
    const TensorAxisLayout& lay,
    const std::vector<TileGraph::TileNode*>& tiles,
    const T* host,
    size_t count,
    TileGraphExecutor& rt)
{
    Index nelems = 1;
    for(Index s : lay.tensor_shape())
    {
        nelems *= s;
    }
    if(count != static_cast<size_t>(nelems))
    {
        throw std::runtime_error(
            "TileGraphExecutor::bind_data: dense size mismatch for logical "
            "tensor");
    }
    const Index vol = lay.grid_volume();
    if(static_cast<Index>(tiles.size()) != vol)
    {
        throw std::runtime_error(
            "TileGraphExecutor::bind_data: tile vector size mismatch");
    }
    std::vector<Index> gc;
    std::vector<Index> local;
    std::vector<Index> global;
    for(Index lin = 0; lin < vol; ++lin)
    {
        lay.grid_coord_from_linear(lin, gc);
        const std::vector<Index> ts = lay.tile_shape_at(gc);
        Index tne = 1;
        for(Index v : ts)
        {
            tne *= v;
        }
        TileGraph::TileNode* tn = tiles[static_cast<size_t>(lin)];
        auto& tile = rt.template get_tile<NntileT>(tn);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index lf = 0; lf < tne; ++lf)
        {
            fortran_tile_linear_to_index(lf, ts, local);
            lay.global_coord(gc, local, global);
            const Index di =
                fortran_dense_linear_index(lay.tensor_shape(), global);
            tile_local[lf] = NntileT(static_cast<CastT>(host[static_cast<size_t>(di)]));
        }
        tile_local.release();
    }
}

template<typename T, typename NntileT, typename CastT>
void gather_logical_tensor(
    const TensorAxisLayout& lay,
    const std::vector<TileGraph::TileNode*>& tiles,
    std::vector<T>& out,
    TileGraphExecutor& rt)
{
    Index nelems = 1;
    for(Index s : lay.tensor_shape())
    {
        nelems *= s;
    }
    out.resize(static_cast<size_t>(nelems));
    const Index vol = lay.grid_volume();
    std::vector<Index> gc;
    std::vector<Index> local;
    std::vector<Index> global;
    for(Index lin = 0; lin < vol; ++lin)
    {
        lay.grid_coord_from_linear(lin, gc);
        const std::vector<Index> ts = lay.tile_shape_at(gc);
        Index tne = 1;
        for(Index v : ts)
        {
            tne *= v;
        }
        TileGraph::TileNode* tn = tiles[static_cast<size_t>(lin)];
        const auto& tile = rt.template get_tile<NntileT>(tn);
        auto tile_local = tile.acquire(STARPU_R);
        for(Index lf = 0; lf < tne; ++lf)
        {
            fortran_tile_linear_to_index(lf, ts, local);
            lay.global_coord(gc, local, global);
            const Index di =
                fortran_dense_linear_index(lay.tensor_shape(), global);
            out[static_cast<size_t>(di)] =
                static_cast<T>(static_cast<CastT>(tile_local[lf]));
        }
        tile_local.release();
    }
}

} // namespace tile_graph_layout_io

template<typename T>
nntile::tile::Tile<T>& TileGraphExecutor::get_tile(const TileNode* node)
{
    auto it = tile_map_.find(node);
    if(it == tile_map_.end())
    {
        throw std::runtime_error(
            "TileGraphExecutor::get_tile: node not found");
    }
    if(node->dtype() != tile_detail::dtype_for<T>::value)
    {
        throw std::runtime_error(
            "TileGraphExecutor::get_tile: wrong type (requested type does "
            "not match tile dtype)");
    }
    auto ptr = std::static_pointer_cast<nntile::tile::Tile<T>>(it->second);
    return *ptr;
}

template<typename T>
void TileGraphExecutor::bind_data(
    TensorGraph::TensorNode const* tensor,
    const T* data,
    size_t count)
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "TileGraphExecutor::bind_data: tensor must be non-null");
    }
    const TileGraph::TensorDescriptor* desc =
        graph_.get_tensor_descriptor(tensor);
    if(desc == nullptr || desc->source_node != tensor)
    {
        throw std::runtime_error(
            "TileGraphExecutor::bind_data: tensor has no TileGraph "
            "descriptor (lower with source_node set)");
    }
    const TensorGraphTiling* tsch = graph_.tiling_scheme();
    const bool use_logical = tsch != nullptr
        && tile_graph_bind_detail::use_logical_layout(desc, tensor);
    if(use_logical)
    {
        const TensorAxisLayout* lay = tsch->find(desc->source_node);
        if(lay == nullptr)
        {
            throw std::runtime_error(
                "TileGraphExecutor::bind_data: missing tiling for tensor '"
                + tensor->name() + "'");
        }
        if(!tile_graph_bind_detail::tensor_desc_has_input_tile(*desc)
            && !tile_graph_bind_detail::tensor_desc_has_output_tile(*desc))
        {
            throw std::runtime_error(
                "bind_data: mark_input(true) or mark_output(true) on tensor '"
                + tensor->name() + "'");
        }
        switch(desc->dtype)
        {
        case DataType::FP32:
            tile_graph_layout_io::scatter_logical_tensor<
                T, nntile::fp32_t, float>(
                *lay, desc->tiles, data, count, *this);
            break;
        case DataType::FP32_FAST_TF32:
            tile_graph_layout_io::scatter_logical_tensor<
                T, nntile::fp32_fast_tf32_t, float>(
                *lay, desc->tiles, data, count, *this);
            break;
        case DataType::FP32_FAST_FP16:
            tile_graph_layout_io::scatter_logical_tensor<
                T, nntile::fp32_fast_fp16_t, float>(
                *lay, desc->tiles, data, count, *this);
            break;
        case DataType::FP32_FAST_BF16:
            tile_graph_layout_io::scatter_logical_tensor<
                T, nntile::fp32_fast_bf16_t, float>(
                *lay, desc->tiles, data, count, *this);
            break;
        case DataType::FP64:
            tile_graph_layout_io::scatter_logical_tensor<
                T, nntile::fp64_t, double>(
                *lay, desc->tiles, data, count, *this);
            break;
        case DataType::FP16:
            tile_graph_layout_io::scatter_logical_tensor<
                T, nntile::fp16_t, float>(
                *lay, desc->tiles, data, count, *this);
            break;
        case DataType::BF16:
            tile_graph_layout_io::scatter_logical_tensor<
                T, nntile::bf16_t, float>(
                *lay, desc->tiles, data, count, *this);
            break;
        case DataType::INT64:
            tile_graph_layout_io::scatter_logical_tensor<
                T, nntile::int64_t, std::int64_t>(
                *lay, desc->tiles, data, count, *this);
            break;
        case DataType::BOOL:
            tile_graph_layout_io::scatter_logical_tensor<
                T, nntile::bool_t, bool>(
                *lay, desc->tiles, data, count, *this);
            break;
        default:
            throw std::runtime_error(
                "TileGraphExecutor::bind_data: unsupported dtype for "
                "logical tensor '" +
                tensor->name() + "'");
        }
        return;
    }
    if(desc->tiles.empty())
    {
        throw std::runtime_error(
            "TileGraphExecutor::bind_data: descriptor has no tiles");
    }
    const std::string& key = desc->tiles[0]->name();
    auto it = runtime_data_.find(key);
    if(it == runtime_data_.end())
    {
        throw std::runtime_error(
            "TileGraphExecutor: data not found: " + key);
    }
    if(!data_is_input_.count(key) && !data_is_output_.count(key))
    {
        throw std::runtime_error(
            "bind_data: tile '" + key +
            "' must be marked as input or output on the data node");
    }
    auto dtype_it = data_dtypes_.find(key);
    if(dtype_it == data_dtypes_.end())
    {
        throw std::runtime_error("bind_data: data dtype not found: " + key);
    }
    DataType dtype = dtype_it->second;
    switch(dtype)
    {
    case DataType::FP32:
        bind_data_impl<T, nntile::fp32_t, float>(key, data, count);
        break;
    case DataType::FP32_FAST_TF32:
        bind_data_impl<T, nntile::fp32_fast_tf32_t, float>(key, data, count);
        break;
    case DataType::FP32_FAST_FP16:
        bind_data_impl<T, nntile::fp32_fast_fp16_t, float>(key, data, count);
        break;
    case DataType::FP32_FAST_BF16:
        bind_data_impl<T, nntile::fp32_fast_bf16_t, float>(key, data, count);
        break;
    case DataType::FP64:
        bind_data_impl<T, nntile::fp64_t, double>(key, data, count);
        break;
    case DataType::FP16:
        bind_data_impl<T, nntile::fp16_t, float>(key, data, count);
        break;
    case DataType::BF16:
        bind_data_impl<T, nntile::bf16_t, float>(key, data, count);
        break;
    case DataType::INT64:
        bind_data_impl<T, nntile::int64_t, std::int64_t>(key, data, count);
        break;
    case DataType::BOOL:
        bind_data_impl<T, nntile::bool_t, bool>(key, data, count);
        break;
    default:
        throw std::runtime_error("Unsupported data type for binding");
    }
}

template<typename T>
void TileGraphExecutor::bind_data(
    TensorGraph::TensorNode const* tensor,
    const std::vector<T>& data)
{
    bind_data(tensor, data.data(), data.size());
}

template<typename T>
void TileGraphExecutor::bind_data(
    NNGraph::TensorNode const* tensor,
    const T* data,
    size_t count)
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "TileGraphExecutor::bind_data: NN tensor must be non-null");
    }
    bind_data(tensor->data(), data, count);
}

template<typename T>
void TileGraphExecutor::bind_data(
    NNGraph::TensorNode const* tensor,
    const std::vector<T>& data)
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "TileGraphExecutor::bind_data: NN tensor must be non-null");
    }
    bind_data(tensor->data(), data);
}

template<typename T>
void TileGraphExecutor::bind_data(
    TileNode const* tile,
    const T* data,
    size_t count)
{
    if(tile == nullptr)
    {
        throw std::invalid_argument(
            "TileGraphExecutor::bind_data: tile must be non-null");
    }
    const std::string& key = tile->name();
    auto it = runtime_data_.find(key);
    if(it == runtime_data_.end())
    {
        throw std::runtime_error(
            "TileGraphExecutor: data not found: " + key);
    }
    if(!data_is_input_.count(key) && !data_is_output_.count(key))
    {
        throw std::runtime_error(
            "bind_data: tile '" + key +
            "' must be marked as input or output on the data node");
    }
    auto dtype_it = data_dtypes_.find(key);
    if(dtype_it == data_dtypes_.end())
    {
        throw std::runtime_error("bind_data: data dtype not found: " + key);
    }
    DataType dtype = dtype_it->second;
    switch(dtype)
    {
    case DataType::FP32:
        bind_data_impl<T, nntile::fp32_t, float>(key, data, count);
        break;
    case DataType::FP32_FAST_TF32:
        bind_data_impl<T, nntile::fp32_fast_tf32_t, float>(key, data, count);
        break;
    case DataType::FP32_FAST_FP16:
        bind_data_impl<T, nntile::fp32_fast_fp16_t, float>(key, data, count);
        break;
    case DataType::FP32_FAST_BF16:
        bind_data_impl<T, nntile::fp32_fast_bf16_t, float>(key, data, count);
        break;
    case DataType::FP64:
        bind_data_impl<T, nntile::fp64_t, double>(key, data, count);
        break;
    case DataType::FP16:
        bind_data_impl<T, nntile::fp16_t, float>(key, data, count);
        break;
    case DataType::BF16:
        bind_data_impl<T, nntile::bf16_t, float>(key, data, count);
        break;
    case DataType::INT64:
        bind_data_impl<T, nntile::int64_t, std::int64_t>(key, data, count);
        break;
    case DataType::BOOL:
        bind_data_impl<T, nntile::bool_t, bool>(key, data, count);
        break;
    default:
        throw std::runtime_error("Unsupported data type for binding");
    }
}

template<typename T>
void TileGraphExecutor::bind_data(
    TileNode const* tile,
    const std::vector<T>& data)
{
    bind_data(tile, data.data(), data.size());
}

template<typename T, typename NntileT, typename CastT>
void TileGraphExecutor::bind_data_impl(
    const std::string& runtime_key,
    const T* data,
    size_t count)
{
    auto& tile = get_data<NntileT>(runtime_key);
    if(count != static_cast<size_t>(tile.nelems))
    {
        throw std::runtime_error(
            "Data size mismatch for data " + runtime_key);
    }
    auto tile_local = tile.acquire(STARPU_W);
    for(size_t i = 0; i < count; ++i)
    {
        tile_local[i] = NntileT(static_cast<CastT>(data[i]));
    }
    tile_local.release();
}

template<typename T>
std::vector<T> TileGraphExecutor::get_output(
    TensorGraph::TensorNode const* tensor)
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "TileGraphExecutor::get_output: tensor must be non-null");
    }
    const TileGraph::TensorDescriptor* desc =
        graph_.get_tensor_descriptor(tensor);
    if(desc == nullptr || desc->source_node != tensor)
    {
        throw std::runtime_error(
            "TileGraphExecutor::get_output: tensor has no TileGraph "
            "descriptor (lower with source_node set)");
    }
    const TensorGraphTiling* tsch = graph_.tiling_scheme();
    const bool use_logical = tsch != nullptr
        && tile_graph_bind_detail::use_logical_layout(desc, tensor);
    if(use_logical)
    {
        const TensorAxisLayout* lay = tsch->find(desc->source_node);
        if(lay == nullptr)
        {
            throw std::runtime_error(
                "TileGraphExecutor::get_output: missing tiling for tensor '"
                + tensor->name() + "'");
        }
        if(!tile_graph_bind_detail::tensor_desc_has_output_tile(*desc))
        {
            throw std::runtime_error(
                "get_output: tensor '" + tensor->name() +
                "' is not marked as output; call mark_output(true) on the "
                "tensor data node");
        }
        std::vector<T> result;
        switch(desc->dtype)
        {
        case DataType::FP32:
            tile_graph_layout_io::gather_logical_tensor<
                T, nntile::fp32_t, float>(
                *lay, desc->tiles, result, *this);
            break;
        case DataType::FP32_FAST_TF32:
            tile_graph_layout_io::gather_logical_tensor<
                T, nntile::fp32_fast_tf32_t, float>(
                *lay, desc->tiles, result, *this);
            break;
        case DataType::FP32_FAST_FP16:
            tile_graph_layout_io::gather_logical_tensor<
                T, nntile::fp32_fast_fp16_t, float>(
                *lay, desc->tiles, result, *this);
            break;
        case DataType::FP32_FAST_BF16:
            tile_graph_layout_io::gather_logical_tensor<
                T, nntile::fp32_fast_bf16_t, float>(
                *lay, desc->tiles, result, *this);
            break;
        case DataType::FP64:
            tile_graph_layout_io::gather_logical_tensor<
                T, nntile::fp64_t, double>(
                *lay, desc->tiles, result, *this);
            break;
        case DataType::FP16:
            tile_graph_layout_io::gather_logical_tensor<
                T, nntile::fp16_t, float>(
                *lay, desc->tiles, result, *this);
            break;
        case DataType::BF16:
            tile_graph_layout_io::gather_logical_tensor<
                T, nntile::bf16_t, float>(
                *lay, desc->tiles, result, *this);
            break;
        case DataType::INT64:
            tile_graph_layout_io::gather_logical_tensor<
                T, nntile::int64_t, std::int64_t>(
                *lay, desc->tiles, result, *this);
            break;
        case DataType::BOOL:
            tile_graph_layout_io::gather_logical_tensor<
                T, nntile::bool_t, bool>(
                *lay, desc->tiles, result, *this);
            break;
        default:
            throw std::runtime_error(
                "TileGraphExecutor::get_output: unsupported dtype for "
                "logical tensor '" +
                tensor->name() + "'");
        }
        return result;
    }
    if(desc->tiles.empty())
    {
        throw std::runtime_error(
            "TileGraphExecutor::get_output: descriptor has no tiles");
    }
    const std::string& key = desc->tiles[0]->name();
    auto data_it = runtime_data_.find(key);
    if(data_it == runtime_data_.end())
    {
        throw std::runtime_error(
            "TileGraphExecutor: data not found: " + key);
    }
    if(!data_is_output_.count(key))
    {
        throw std::runtime_error(
            "get_output: tile '" + key +
            "' is not marked as output; call mark_output(true) on the data "
            "node");
    }
    auto dtype_it = data_dtypes_.find(key);
    if(dtype_it == data_dtypes_.end())
    {
        throw std::runtime_error("Data dtype not found: " + key);
    }
    DataType dtype = dtype_it->second;
    std::vector<T> result;
    switch(dtype)
    {
    case DataType::FP32:
        get_output_impl<T, nntile::fp32_t, float>(key, result);
        break;
    case DataType::FP32_FAST_TF32:
        get_output_impl<T, nntile::fp32_fast_tf32_t, float>(key, result);
        break;
    case DataType::FP32_FAST_FP16:
        get_output_impl<T, nntile::fp32_fast_fp16_t, float>(key, result);
        break;
    case DataType::FP32_FAST_BF16:
        get_output_impl<T, nntile::fp32_fast_bf16_t, float>(key, result);
        break;
    case DataType::FP64:
        get_output_impl<T, nntile::fp64_t, double>(key, result);
        break;
    case DataType::FP16:
        get_output_impl<T, nntile::fp16_t, float>(key, result);
        break;
    case DataType::BF16:
        get_output_impl<T, nntile::bf16_t, float>(key, result);
        break;
    case DataType::INT64:
        get_output_impl<T, nntile::int64_t, std::int64_t>(key, result);
        break;
    case DataType::BOOL:
        get_output_impl<T, nntile::bool_t, bool>(key, result);
        break;
    default:
        throw std::runtime_error("Unsupported data type for get_output");
    }
    return result;
}

template<typename T>
std::vector<T> TileGraphExecutor::get_output(
    NNGraph::TensorNode const* tensor)
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "TileGraphExecutor::get_output: NN tensor must be non-null");
    }
    return get_output<T>(tensor->data());
}

template<typename T>
std::vector<T> TileGraphExecutor::get_output(TileNode const* tile)
{
    if(tile == nullptr)
    {
        throw std::invalid_argument(
            "TileGraphExecutor::get_output: tile must be non-null");
    }
    if(!tile->is_output())
    {
        throw std::runtime_error(
            "get_output: tile must be marked output on the data node");
    }
    const std::string& key = tile->name();
    auto data_it = runtime_data_.find(key);
    if(data_it == runtime_data_.end())
    {
        throw std::runtime_error(
            "TileGraphExecutor: data not found: " + key);
    }
    auto dtype_it = data_dtypes_.find(key);
    if(dtype_it == data_dtypes_.end())
    {
        throw std::runtime_error("Data dtype not found: " + key);
    }
    DataType dtype = dtype_it->second;
    std::vector<T> result;
    switch(dtype)
    {
    case DataType::FP32:
        get_output_impl<T, nntile::fp32_t, float>(key, result);
        break;
    case DataType::FP32_FAST_TF32:
        get_output_impl<T, nntile::fp32_fast_tf32_t, float>(key, result);
        break;
    case DataType::FP32_FAST_FP16:
        get_output_impl<T, nntile::fp32_fast_fp16_t, float>(key, result);
        break;
    case DataType::FP32_FAST_BF16:
        get_output_impl<T, nntile::fp32_fast_bf16_t, float>(key, result);
        break;
    case DataType::FP64:
        get_output_impl<T, nntile::fp64_t, double>(key, result);
        break;
    case DataType::FP16:
        get_output_impl<T, nntile::fp16_t, float>(key, result);
        break;
    case DataType::BF16:
        get_output_impl<T, nntile::bf16_t, float>(key, result);
        break;
    case DataType::INT64:
        get_output_impl<T, nntile::int64_t, std::int64_t>(key, result);
        break;
    case DataType::BOOL:
        get_output_impl<T, nntile::bool_t, bool>(key, result);
        break;
    default:
        throw std::runtime_error("Unsupported data type for get_output");
    }
    return result;
}

template<typename T, typename NntileT, typename CastT>
void TileGraphExecutor::get_output_impl(const std::string& runtime_key,
                                         std::vector<T>& result)
{
    auto& tile = get_data<NntileT>(runtime_key);
    result.resize(tile.nelems);
    auto tile_local = tile.acquire(STARPU_R);
    for(Index i = 0; i < tile.nelems; ++i)
    {
        result[i] = static_cast<T>(static_cast<CastT>(tile_local[i]));
    }
    tile_local.release();
}

inline DataType TileGraphExecutor::get_dtype(
    NNGraph::TensorNode const* tensor) const
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "TileGraphExecutor::get_dtype: NN tensor must be non-null");
    }
    return get_dtype(tensor->data());
}

} // namespace nntile::graph
