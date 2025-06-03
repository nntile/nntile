/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/tensor.hh
 * Tensor<T> class
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdlib>
#include <nntile/tensor/traits.hh>
#include <nntile/tile/tile.hh>
#include <nntile/starpu/codelet.hh>
//#include <starpu_mpi.h>
#include <starpu.h>
#include <nntile/starpu/accumulate.hh>
#include <nntile/starpu/accumulate_hypot.hh>
#include <nntile/starpu/accumulate_maxsumexp.hh>
#include <nntile/starpu/clear.hh>

#define starpu_mpi_tag_t std::int64_t

namespace nntile::tensor
{

//! Many-dimensional tensor, presented by a set of subtensors (tiles)
//
// This is the main data storage class, that assumes a tensor as a set of
// tiles, handled by StarPU runtime system.
template<typename T>
class Tensor: public TensorTraits
{
public:
    //! Traits of all tiles
    std::vector<tile::TileTraits> tile_traits;
    //! StarPU handles of all tiles
    std::vector<starpu::VariableHandle> tile_handles;
    //! Distribution of tiles
    std::vector<int> tile_distr;
    //! Flag to really use wont_use
    int wont_use_flag;

    //! Constructor
    explicit Tensor(
            const TensorTraits &traits,
            const std::vector<int> &distribution=std::vector<int>(),
            const char *name=nullptr
        ):
        TensorTraits(traits),
        tile_distr(distribution),
        wont_use_flag(0)
    {
        // Check if distribution is empty
        if(tile_distr.size() == 0)
        {
            // Define it as a vector of zeros for now. In far future, when we
            // will have MPI support, we will need to use certain distribution
            // strategy
            tile_distr = std::vector<int>(grid.nelems, 0);
        }
        // Check distribution
        if(tile_distr.size() != grid.nelems)
        {
            throw std::runtime_error("Wrong distribution");
        }
        // Register tiles
        tile_traits.reserve(grid.nelems);
        tile_handles.reserve(grid.nelems);
        for(Index i = 0; i < grid.nelems; ++i)
        {
            // Get tile index
            const auto tile_index = grid.linear_to_index(i);
            // Get shape of corresponding tile
            const auto tile_shape = TensorTraits::get_tile_shape(tile_index);
            // Generate traits for the tile
            tile_traits.emplace_back(tile_shape);
            // Set StarPU-managed handle
            tile_handles.emplace_back(sizeof(T)*tile_traits[i].nelems);
            // Set coordinate of the tile
            starpu_data_set_coordinates(
                tile_handles[i].get(),
                tile_index.size(),
                tile_index.data()
            );
            // Set name of the tile the same as the tensor name
            if(name != nullptr)
            {
                starpu_data_set_name(
                    tile_handles[i].get(),
                    name
                );
            }
            // Disable Out-of-Core by default
            starpu_data_set_ooc_flag(
                tile_handles[i].get(),
                0
            );
        }
    }

    //! Destructor unregisters all tiles in an async manner
    ~Tensor()
    {
        unregister_submit();
    }

    //! Get tile by linear offset
    tile::Tile<T> get_tile(Index linear_offset) const
    {
        if(linear_offset < 0 or linear_offset >= grid.nelems)
        {
            throw std::runtime_error("Tile offset is out of bounds");
        }
        return tile::Tile<T>(tile_traits[linear_offset],
                tile_handles[linear_offset]);
    }
    tile::Tile<T> get_tile(const std::vector<Index> &tile_index) const
    {
        Index linear_offset = grid.index_to_linear(tile_index);
        return tile::Tile<T>(tile_traits[linear_offset],
                tile_handles[linear_offset]);
    }
    const tile::TileTraits &get_tile_traits(Index linear_offset) const
    {
        return tile_traits[linear_offset];
    }
    const tile::TileTraits &get_tile_traits(
            const std::vector<Index> &tile_index) const
    {
        Index linear_offset = grid.index_to_linear(tile_index);
        return tile_traits[linear_offset];
    }
    const starpu::Handle &get_tile_handle(Index linear_offset) const
    {
        return tile_handles[linear_offset];
    }
    const starpu::Handle &get_tile_handle(
            const std::vector<Index> &tile_index) const
    {
        Index linear_offset = grid.index_to_linear(tile_index);
        return tile_handles[linear_offset];
    }
    //! Unregister underlying handles in a blocking manner
    void unregister()
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            tile_handles[i].unregister();
        }
    }
    //! Unregister underlying handles in an async manner
    void unregister_submit()
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            tile_handles[i].unregister_submit();
        }
    }
    //! Unregister underlying handles in a blocking manner without coherency
    void unregister_no_coherency()
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            tile_handles[i].unregister_no_coherency();
        }
    }
    //! Invalidate tensor values in an async manner
    void invalidate_submit() const
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            auto tmp = tile_handles[i].get();
            starpu_data_invalidate_submit(tmp);
        }
    }
    //! Advice to evict data from GPU
    void wont_use() const
    {
        // Form a list of tiles to be evicted from GPU with help of starpu_data_wont_use
        for(Index i = 0; i < grid.nelems; ++i)
        {
            auto tmp = tile_handles[i].get();
            // Do wont_use only if we enforce offloading to RAM or Disk
            if(wont_use_flag == 1 or starpu_data_get_ooc_flag(tmp) == 1)
            {
                // Advise to offload the data to disk
                starpu_data_wont_use(tmp);
                // Clear out the data from the GPU nodes
                for(int node = 0; node < STARPU_MAXNODES; ++node)
                {
                    if(starpu_node_get_kind(node) == STARPU_CUDA_RAM
                        and starpu_data_can_evict(tmp, node, STARPU_FETCH))
                    {
                        starpu_data_evict_from_node(tmp, node);
                    }
                }
            }
            // Evict also from RAM if Disk is enabled
            if(starpu_data_get_ooc_flag(tmp) == 1)
            {
                for(int node = 0; node < STARPU_MAXNODES; ++node)
                {
                    if(starpu_node_get_kind(node) == STARPU_CPU_RAM
                        and starpu_data_can_evict(tmp, node, STARPU_FETCH))
                    {
                        starpu_data_evict_from_node(tmp, node);
                    }
                }
            }
        }
    }
    //! Flush tensor from MPI caches
    void mpi_flush() const
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            auto tmp = tile_handles[i].get();
            //starpu_mpi_cache_flush(MPI_COMM_WORLD, tmp);
        }
    }
    //! Set reduction function for addition
    void set_reduction_add() const
    {
        // Only do something if T is a floating point type
        if constexpr (is_floating_point_type<T>)
        {
            // Set reduction function for addition
            for(Index i = 0; i < grid.nelems; ++i)
            {
                auto tmp = tile_handles[i].get();
                auto accumulate_pack = nntile::starpu::accumulate;
                auto accumulate_op = static_cast<starpu::Accumulate<std::tuple<T>>>(accumulate_pack);
                auto clear_codelet = &nntile::starpu::clear.codelet;
                starpu_data_set_reduction_methods(
                    tmp,
                    &accumulate_op.codelet,
                    clear_codelet
                );
            }
        }
    }
    //! Set reduction function for hypot
    void set_reduction_hypot() const
    {
        // Only do something if T is a floating point type
        if constexpr (is_floating_point_type<T>)
        {
            // Set reduction function for hypot
            for(Index i = 0; i < grid.nelems; ++i)
            {
                auto tmp = tile_handles[i].get();
                auto accumulate_pack = nntile::starpu::accumulate_hypot;
                auto accumulate_op = static_cast<starpu::AccumulateHypot<std::tuple<T>>>(accumulate_pack);
                auto clear_codelet = &nntile::starpu::clear.codelet;
                starpu_data_set_reduction_methods(
                    tmp,
                    &accumulate_op.codelet,
                    clear_codelet
                );
            }
        }
    }
    //! Set reduction function for maxsumexp
    void set_reduction_maxsumexp() const
    {
        // Only do something if T is a floating point type
        if constexpr (is_floating_point_type<T>)
        {
            // Set reduction function for maxsumexp
            for(Index i = 0; i < grid.nelems; ++i)
            {
                auto tmp = tile_handles[i].get();
                auto accumulate_pack = nntile::starpu::accumulate_maxsumexp;
                auto accumulate_op = static_cast<starpu::AccumulateMaxSumExp<std::tuple<T>>>(accumulate_pack);
                auto clear_codelet = &(nntile::starpu::clear.codelet);
                starpu_data_set_reduction_methods(
                    tmp,
                    &accumulate_op.codelet,
                    clear_codelet
                );
            }
        }
    }
    //! Print scalar tensor asynchronously
    void print_scalar_async() const
    {
        if(ndim != 0)
        {
            throw std::runtime_error("Only scalar tensors can be printed");
        }
        auto handle = tile_handles[0].get();
        void **args = reinterpret_cast<void **>(std::malloc(sizeof(*args)));
        *args = reinterpret_cast<void *>(handle);
        int ret = starpu_data_acquire_cb(handle, STARPU_R,
                reinterpret_cast<void (*)(void *)>(&_print_scalar_async_helper),
                reinterpret_cast<void *>(args));
        if(ret != 0)
        {
            throw std::runtime_error("Error in starpu_data_acquire_cb()");
        }
    }
    //! Helper for async printing
    static void _print_scalar_async_helper(void *args)
    {
        std::cerr << "IN CALLBACK args " << args << "\n";
        auto handle = *reinterpret_cast<starpu_data_handle_t *>(args);
        std::cerr << "IN CALLBACK handle " << handle << "\n";
        T *data = reinterpret_cast<T *>(starpu_data_get_local_ptr(handle));
        std::cout << "Value: " << *data << "\n";
        starpu_data_release(handle);
    }
    //! Get size of the data in bytes
    size_t get_nbytes()
    {
        return sizeof(T) * nelems;
    }
    //! Enable data offloading to RAM when wont_use is called
    void force_offload_ram_enable()
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            auto tmp = tile_handles[i].get();
            // Set write-through mask to enable offloading after each update
            starpu_data_set_wt_mask(tmp, 1 << STARPU_MAIN_RAM);
        }
        wont_use_flag = 1;
    }
    //! Disable data offloading to RAM when wont_use is called
    void force_offload_ram_disable()
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            auto tmp = tile_handles[i].get();
            // Set write-through mask to disable offloading after each update
            starpu_data_set_wt_mask(tmp, 0);
        }
        wont_use_flag = 0;
    }
    //! Enforce data offloading to disk when wont_use is called
    void force_offload_disk_enable()
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            auto tmp = tile_handles[i].get();
            starpu_data_set_ooc_flag(tmp, 1);
        }
    }
    //! Disable data offloading to disk when wont_use is called
    void force_offload_disk_disable()
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            auto tmp = tile_handles[i].get();
            starpu_data_set_ooc_flag(tmp, 0);
        }
    }
};

} // namespace nntile::tensor
