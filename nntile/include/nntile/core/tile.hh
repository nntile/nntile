/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/core/tile.hh
 * Tile<T> class
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/traits.hh>
#include <nntile/defs.h>

#ifdef NNTILE_USE_NNHAUL

#include <nntile/data_access.hh>

#include <nnhaul/nnhaul.hh>

#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace nntile::core
{

template<typename T>
class TileLocalData;

template<typename T>
class Tile: public TileTraits
{
    struct Payload
    {
        std::unique_ptr<::nnhaul::Handle> handle;
        bool live = true;
    };

    std::shared_ptr<Payload> payload_;

    static std::size_t storage_bytes(Index nelems_)
    {
        std::size_t const size =
            static_cast<std::size_t>(nelems_) * sizeof(T);
        if (size / sizeof(T) != static_cast<std::size_t>(nelems_))
        {
            throw std::runtime_error(
                "Type size_t is not enough to hold size of provided buffer");
        }
        return size;
    }

    void alloc_handle()
    {
        payload_ = std::make_shared<Payload>();
        payload_->handle =
            std::make_unique<::nnhaul::Handle>(storage_bytes(nelems));
    }

public:
    Tile(TileTraits const &traits_, Tile const &other):
        TileTraits(traits_),
        payload_(other.payload_)
    {
    }

    explicit Tile(std::vector<Index> const &shape_):
        TileTraits(shape_)
    {
        alloc_handle();
    }

    explicit Tile(TileTraits const &traits):
        TileTraits(traits)
    {
        alloc_handle();
    }

    Tile(std::vector<Index> const &shape_, T *ptr, Index ptr_nelems):
        TileTraits(shape_)
    {
        if (nelems > ptr_nelems)
        {
            throw std::runtime_error(
                "Required memory size is larger than actually "
                "allocated memory");
        }
        alloc_handle();
        {
            ::nnhaul::Acquired local =
                payload_->handle->acquire(::nnhaul::Access::Write);
            std::memcpy(local.get_ptr<T>(), ptr, storage_bytes(nelems));
        }
    }

    Tile(TileTraits const &traits, T *ptr, Index ptr_nelems):
        Tile(traits.shape, ptr, ptr_nelems)
    {
    }

    ::nnhaul::Handle *get() const
    {
        return payload_ ? payload_->handle.get() : nullptr;
    }

    operator ::nnhaul::Handle &() const
    {
        return *payload_->handle;
    }

    ::nnhaul::Handle &handle() const
    {
        return *payload_->handle;
    }

    int mpi_get_rank() const
    {
        return 0;
    }

    void mpi_transfer(int, int) const
    {
    }

    void invalidate_submit() const
    {
        if (!payload_ || !payload_->live)
        {
            return;
        }
        payload_->handle->invalidate();
        payload_->live = false;
    }

    void unregister_submit() const
    {
        invalidate_submit();
    }

    TileLocalData<T> acquire(int mode) const;

    TileLocalData<T> acquire_async(int mode) const;
};

template<typename T>
class TileLocalData
{
    Tile<T> const *tile_ = nullptr;
    std::optional<::nnhaul::Acquired> held_;

    static ::nnhaul::Access map_mode(int mode)
    {
        if (mode == STARPU_R)
        {
            return ::nnhaul::Access::Read;
        }
        if (mode == STARPU_W)
        {
            return ::nnhaul::Access::Write;
        }
        if (mode == STARPU_RW)
        {
            return ::nnhaul::Access::ReadWrite;
        }
        throw std::runtime_error("redux or commute acquire is not supported");
    }

public:
    TileLocalData(Tile<T> const &tile, int mode, bool is_blocking = true):
        tile_(&tile)
    {
        if (is_blocking)
        {
            acquire(mode);
        }
    }

    TileLocalData(TileLocalData &&) noexcept = default;
    TileLocalData &operator=(TileLocalData &&) noexcept = default;
    TileLocalData(TileLocalData const &) = delete;
    TileLocalData &operator=(TileLocalData const &) = delete;

    void acquire(int mode)
    {
        release();
        held_.emplace(tile_->handle().acquire(map_mode(mode)));
    }

    void release()
    {
        held_.reset();
    }

    T const &operator[](Index i) const
    {
        return get_ptr()[i];
    }

    T &operator[](Index i)
    {
        return get_ptr()[i];
    }

    T *get_ptr() const
    {
        return held_->template get_ptr<T>();
    }
};

template<typename T>
TileLocalData<T> Tile<T>::acquire(int mode) const
{
    return TileLocalData<T>(*this, mode, true);
}

template<typename T>
TileLocalData<T> Tile<T>::acquire_async(int mode) const
{
    return TileLocalData<T>(*this, mode, true);
}

} // namespace nntile::core

#else

#include <nntile/starpu/handle.hh>

namespace nntile::core
{

template<typename T>
class TileLocalData;

template<typename T>
class Tile: public TileTraits, public starpu::VariableHandle
{
    size_t _get_size(Index ptr_nelems)
    {
        if (nelems > ptr_nelems)
        {
            throw std::runtime_error(
                "Required memory size is larger than actually "
                "allocated memory");
        }
        std::size_t size = nelems * sizeof(T);
        if (size / sizeof(T) != static_cast<std::size_t>(nelems))
        {
            throw std::runtime_error(
                "Type size_t is not enough to hold size of provided buffer");
        }
        return size;
    }

public:
    Tile(TileTraits const &traits_, starpu::VariableHandle const &handle_):
        TileTraits(traits_),
        starpu::VariableHandle(handle_)
    {
    }

    //! Allocation size of a freshly owned tile. Only the TileTraits
    //! base is initialized when this runs, but nelems is already
    //! validated there; the round-trip check mirrors _get_size.
    size_t _get_alloc_size()
    {
        size_t size = static_cast<size_t>(nelems) * sizeof(T);
        if(size / sizeof(T) != static_cast<size_t>(nelems))
        {
            throw std::runtime_error(
                "Type size_t is not enough to hold size of the tile");
        }
        return size;
    }

    explicit Tile(std::vector<Index> const &shape_):
        TileTraits(shape_),
        starpu::VariableHandle(_get_alloc_size())
    {
    }

    explicit Tile(TileTraits const &traits):
        TileTraits(traits),
        starpu::VariableHandle(_get_alloc_size())
    {
    }

    Tile(std::vector<Index> const &shape_, T *ptr, Index ptr_nelems):
        TileTraits(shape_),
        starpu::VariableHandle(ptr, _get_size(ptr_nelems))
    {
    }

    Tile(TileTraits const &traits, T *ptr, Index ptr_nelems):
        TileTraits(traits),
        starpu::VariableHandle(ptr, _get_size(ptr_nelems))
    {
    }

    TileLocalData<T> acquire(starpu_data_access_mode mode) const;

    TileLocalData<T> acquire_async(starpu_data_access_mode mode) const;
};

template<typename T>
class TileLocalData: public starpu::HandleLocalData
{
public:
    TileLocalData(
        Tile<T> const &tile,
        starpu_data_access_mode mode,
        bool is_blocking = true):
        starpu::HandleLocalData(tile, mode, is_blocking)
    {
    }

    T const &operator[](Index i) const
    {
        return get_ptr()[i];
    }

    T &operator[](Index i)
    {
        return get_ptr()[i];
    }

    T *get_ptr() const
    {
        return reinterpret_cast<T *>(starpu::HandleLocalData::get_ptr());
    }
};

template<typename T>
TileLocalData<T> Tile<T>::acquire(starpu_data_access_mode mode) const
{
    return TileLocalData<T>(*this, mode, true);
}

template<typename T>
TileLocalData<T> Tile<T>::acquire_async(starpu_data_access_mode mode) const
{
    return TileLocalData<T>(*this, mode, false);
}

} // namespace nntile::core

#endif
