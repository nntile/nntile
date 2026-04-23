/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/graph_runtime.hh
 * TileGraph::Runtime - runtime execution of a TileGraph.
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
#include <nntile/graph/tensor/tensor_graph_tiling.hh>
#include <nntile/graph/tile/graph_data_node.hh>
#include <nntile/tile/tile.hh>

namespace nntile::graph
{

//! Runtime for executing a TileGraph.
//! compile() allocates tiles and builds execution order; execute() runs ops.
class TileGraph::Runtime
{
public:
    using TileNode = TileGraph::TileNode;
    using OpNode = TileGraph::OpNode;

    explicit Runtime(const TileGraph& graph);

    void compile();

    template<typename T>
    void bind_data(const std::string& name, const T* data, size_t count);

    template<typename T>
    void bind_data(const std::string& name, const std::vector<T>& data);

    void execute();

    void wait();

    template<typename T>
    std::vector<T> get_output(const std::string& name);

    template<typename T>
    nntile::tile::Tile<T>& get_data(const std::string& name);

    template<typename T>
    nntile::tile::Tile<T>& get_tile(const TileNode* node);

    DataType get_dtype(const std::string& name) const
    {
        return data_dtypes_.at(name);
    }

    DataType get_dtype(const TileNode* node) const
    {
        return node->dtype();
    }

    bool is_compiled() const { return compiled_; }

private:
    void allocate_impl();
    void eliminate_dead_ops();
    void invalidate_data(const std::string& name);
    void invalidate_unused_tiles(size_t op_idx);

    template<typename T, typename NntileT, typename CastT>
    void bind_data_impl(const std::string& name, const T* data, size_t count);
    template<typename T, typename NntileT, typename CastT>
    void get_output_impl(const std::string& name, std::vector<T>& result);

    const TileGraph& graph_;
    std::map<const TileNode*, std::shared_ptr<void>> tile_map_;
    std::map<std::string, std::shared_ptr<void>> runtime_data_;
    std::map<std::string, DataType> data_dtypes_;
    std::vector<std::shared_ptr<OpNode>> execution_order_;
    std::set<std::string> data_is_input_;
    std::set<std::string> data_is_output_;
    std::map<std::string, size_t> data_last_use_;
    bool compiled_ = false;
};

// ---------------------------------------------------------------------------
// Template implementation
// ---------------------------------------------------------------------------

template<typename T>
nntile::tile::Tile<T>& TileGraph::Runtime::get_data(const std::string& name)
{
    auto it = runtime_data_.find(name);
    if(it == runtime_data_.end())
    {
        throw std::runtime_error("TileGraph::Runtime: data not found: " + name);
    }
    return *static_cast<nntile::tile::Tile<T>*>(it->second.get());
}

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
    TileGraph::Runtime& rt)
{
    Index nelems = 1;
    for(Index s : lay.tensor_shape())
    {
        nelems *= s;
    }
    if(count != static_cast<size_t>(nelems))
    {
        throw std::runtime_error(
            "TileGraph::Runtime::bind_data: dense size mismatch for logical "
            "tensor");
    }
    const Index vol = lay.grid_volume();
    if(static_cast<Index>(tiles.size()) != vol)
    {
        throw std::runtime_error(
            "TileGraph::Runtime::bind_data: tile vector size mismatch");
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
    TileGraph::Runtime& rt)
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
nntile::tile::Tile<T>& TileGraph::Runtime::get_tile(const TileNode* node)
{
    auto it = tile_map_.find(node);
    if(it == tile_map_.end())
    {
        throw std::runtime_error(
            "TileGraph::Runtime::get_tile: node not found");
    }
    if(node->dtype() != tile_detail::dtype_for<T>::value)
    {
        throw std::runtime_error(
            "TileGraph::Runtime::get_tile: wrong type (requested type does "
            "not match tile dtype)");
    }
    auto ptr = std::static_pointer_cast<nntile::tile::Tile<T>>(it->second);
    return *ptr;
}

template<typename T>
void TileGraph::Runtime::bind_data(const std::string& name, const T* data,
                                   size_t count)
{
    const TensorGraphTiling* tsch = graph_.tiling_scheme();
    if(tsch != nullptr)
    {
        const TileGraph::TensorDescriptor* desc =
            graph_.get_tensor_descriptor(name);
        if(desc != nullptr && desc->source_node != nullptr)
        {
            const bool use_logical =
                desc->tiles.size() > static_cast<size_t>(1) ||
                (desc->tiles.size() == static_cast<size_t>(1) &&
                 desc->tiles[0]->name() != name);
            if(use_logical)
            {
                const TensorAxisLayout* lay = tsch->find(desc->source_node);
                if(lay == nullptr)
                {
                    throw std::runtime_error(
                        "TileGraph::Runtime::bind_data: missing tiling for '" +
                        name + "'");
                }
                if(!data_is_input_.count(name) && !data_is_output_.count(name))
                {
                    throw std::runtime_error(
                        "bind_data: data '" + name +
                        "' must be marked as input or output; "
                        "call mark_input(true) or mark_output(true) on the "
                        "tensor data node");
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
                        "TileGraph::Runtime::bind_data: unsupported dtype for "
                        "logical tensor '" +
                        name + "'");
                }
                return;
            }
        }
    }

    auto it = runtime_data_.find(name);
    if(it == runtime_data_.end())
    {
        throw std::runtime_error("TileGraph::Runtime: data not found: " + name);
    }
    if(!data_is_input_.count(name) && !data_is_output_.count(name))
    {
        throw std::runtime_error(
            "bind_data: data '" + name +
            "' must be marked as input or output; "
            "call mark_input(true) or mark_output(true) on the data node");
    }

    auto dtype_it = data_dtypes_.find(name);
    if(dtype_it == data_dtypes_.end())
    {
        throw std::runtime_error("bind_data: data dtype not found: " + name);
    }
    DataType dtype = dtype_it->second;

    switch(dtype)
    {
        case DataType::FP32:
            bind_data_impl<T, nntile::fp32_t, float>(name, data, count);
            break;
        case DataType::FP32_FAST_TF32:
            bind_data_impl<T, nntile::fp32_fast_tf32_t, float>(name, data, count);
            break;
        case DataType::FP32_FAST_FP16:
            bind_data_impl<T, nntile::fp32_fast_fp16_t, float>(name, data, count);
            break;
        case DataType::FP32_FAST_BF16:
            bind_data_impl<T, nntile::fp32_fast_bf16_t, float>(name, data, count);
            break;
        case DataType::FP64:
            bind_data_impl<T, nntile::fp64_t, double>(name, data, count);
            break;
        case DataType::FP16:
            bind_data_impl<T, nntile::fp16_t, float>(name, data, count);
            break;
        case DataType::BF16:
            bind_data_impl<T, nntile::bf16_t, float>(name, data, count);
            break;
        case DataType::INT64:
            bind_data_impl<T, nntile::int64_t, std::int64_t>(name, data, count);
            break;
        case DataType::BOOL:
            bind_data_impl<T, nntile::bool_t, bool>(name, data, count);
            break;
        default:
            throw std::runtime_error("Unsupported data type for binding");
    }
}

template<typename T>
void TileGraph::Runtime::bind_data(const std::string& name,
                                   const std::vector<T>& data)
{
    bind_data(name, data.data(), data.size());
}

template<typename T, typename NntileT, typename CastT>
void TileGraph::Runtime::bind_data_impl(const std::string& name,
                                        const T* data, size_t count)
{
    auto& tile = get_data<NntileT>(name);
    if(count != static_cast<size_t>(tile.nelems))
    {
        throw std::runtime_error("Data size mismatch for data " + name);
    }
    auto tile_local = tile.acquire(STARPU_W);
    for(size_t i = 0; i < count; ++i)
    {
        tile_local[i] = NntileT(static_cast<CastT>(data[i]));
    }
    tile_local.release();
}

template<typename T>
std::vector<T> TileGraph::Runtime::get_output(const std::string& name)
{
    const TensorGraphTiling* tsch = graph_.tiling_scheme();
    if(tsch != nullptr)
    {
        const TileGraph::TensorDescriptor* desc =
            graph_.get_tensor_descriptor(name);
        if(desc != nullptr && desc->source_node != nullptr)
        {
            const bool use_logical =
                desc->tiles.size() > static_cast<size_t>(1) ||
                (desc->tiles.size() == static_cast<size_t>(1) &&
                 desc->tiles[0]->name() != name);
            if(use_logical)
            {
                const TensorAxisLayout* lay = tsch->find(desc->source_node);
                if(lay == nullptr)
                {
                    throw std::runtime_error(
                        "TileGraph::Runtime::get_output: missing tiling for '" +
                        name + "'");
                }
                if(!data_is_output_.count(name))
                {
                    throw std::runtime_error(
                        "get_output: data '" + name +
                        "' is not marked as output; intermediate tiles are "
                        "invalidated during execution; call mark_output(true) on "
                        "the tensor data node");
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
                        "TileGraph::Runtime::get_output: unsupported dtype for "
                        "logical tensor '" +
                        name + "'");
                }
                return result;
            }
        }
    }

    auto data_it = runtime_data_.find(name);
    if(data_it == runtime_data_.end())
    {
        throw std::runtime_error("TileGraph::Runtime: data not found: " + name);
    }
    if(!data_is_output_.count(name))
    {
        throw std::runtime_error(
            "get_output: data '" + name +
            "' is not marked as output; intermediate tiles are invalidated "
            "during execution; call mark_output(true) on the data node");
    }
    auto dtype_it = data_dtypes_.find(name);
    if(dtype_it == data_dtypes_.end())
    {
        throw std::runtime_error("Data dtype not found: " + name);
    }
    DataType dtype = dtype_it->second;
    std::vector<T> result;

    switch(dtype)
    {
        case DataType::FP32:
            get_output_impl<T, nntile::fp32_t, float>(name, result);
            break;
        case DataType::FP32_FAST_TF32:
            get_output_impl<T, nntile::fp32_fast_tf32_t, float>(name, result);
            break;
        case DataType::FP32_FAST_FP16:
            get_output_impl<T, nntile::fp32_fast_fp16_t, float>(name, result);
            break;
        case DataType::FP32_FAST_BF16:
            get_output_impl<T, nntile::fp32_fast_bf16_t, float>(name, result);
            break;
        case DataType::FP64:
            get_output_impl<T, nntile::fp64_t, double>(name, result);
            break;
        case DataType::FP16:
            get_output_impl<T, nntile::fp16_t, float>(name, result);
            break;
        case DataType::BF16:
            get_output_impl<T, nntile::bf16_t, float>(name, result);
            break;
        case DataType::INT64:
            get_output_impl<T, nntile::int64_t, std::int64_t>(name, result);
            break;
        case DataType::BOOL:
            get_output_impl<T, nntile::bool_t, bool>(name, result);
            break;
        default:
            throw std::runtime_error("Unsupported data type for get_output");
    }

    return result;
}

template<typename T, typename NntileT, typename CastT>
void TileGraph::Runtime::get_output_impl(const std::string& name,
                                         std::vector<T>& result)
{
    auto& tile = get_data<NntileT>(name);
    result.resize(tile.nelems);
    auto tile_local = tile.acquire(STARPU_R);
    for(Index i = 0; i < tile.nelems; ++i)
    {
        result[i] = static_cast<T>(static_cast<CastT>(tile_local[i]));
    }
    tile_local.release();
}

} // namespace nntile::graph
