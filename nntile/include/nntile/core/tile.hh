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


#include <nntile/data_access.hh>
#include <nntile/nnhaul/chunk_gate.hh>
#include <nntile/nnhaul/stage.hh>

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

        ~Payload()
        {
            // Unregister before the Handle member dies so a recycled
            // address can never alias a stale registry entry.
            if (handle)
            {
                ::nntile::nnhaul::stage::forget_handle(*handle);
            }
        }
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
        std::size_t const nbytes = storage_bytes(nelems);
        // Classic-stack chunked tiles: one chunk class under the new
        // nnhaul engine (every tile decomposes into whole class chunks).
        // Always on: the NNHaul product has no unchunked tiles.
        std::size_t const chunk_bytes = ::nntile::nnhaul::tile_chunking_enabled()
            ? ::nnhaul::chunk_class_bytes()
            : 0;
        payload_->handle =
            std::make_unique<::nnhaul::Handle>(nbytes, chunk_bytes);
        if (chunk_bytes != 0)
        {
            // Exact chunkedness for the stage helper and the compile-time
            // torch gate (raw handles cannot be queried at submit time).
            ::nntile::nnhaul::stage::note_chunked(*payload_->handle,
                chunk_bytes);
        }
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
        if (mode == ACCESS_R)
        {
            return ::nnhaul::Access::Read;
        }
        if (mode == ACCESS_W)
        {
            return ::nnhaul::Access::Write;
        }
        if (mode == ACCESS_RW)
        {
            return ::nnhaul::Access::ReadWrite;
        }
        throw std::runtime_error("commute acquire is not supported");
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
