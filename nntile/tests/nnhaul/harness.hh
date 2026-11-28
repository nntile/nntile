/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/harness.hh
 * Shared harness for the kernel-versus-wrapper tests: runtime guard, tile
 * helpers, and numeric comparison against reference CPU kernels.
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/defs.h>
#include <nntile/base_types.hh>
#include <nntile/context.hh>
#include <nntile/nnhaul/stage.hh>

#include <nnhaul/nnhaul.hh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace wrap
{

inline void cuda_ok(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

// Context(0, 1) is the rolling runtime's shape: one CUDA worker, no
// CPU workers. Every submit lands on worker 0 (the CUDA worker); the
// "cpu" legs compare against host reference kernels, not CPU workers.
inline constexpr int cuda_worker = 0;
inline constexpr int cpu_worker = 0;

struct Session
{
    // The host pool bound passed to nntile::nnhaul::init. The engine sizes
    // its pinned host pool from cpu_cap_bytes at init (one
    // cudaHostAlloc of the whole bound), so the default nntile
    // computes (half of physical RAM) would try to pin hundreds of
    // GiB and wedge the driver. The wrapper tests stage a few MiB;
    // 2 GiB is a whole-class carve with ample margin.
    static constexpr std::size_t host_cap_bytes = std::size_t{2} << 30;

    // The per-device arena cap. With the default (free - 10%) the
    // engine's arena plus its 1 GiB copy workspace consume nearly
    // the whole GPU, and the direct-kernel comparison legs of these
    // tests then fail their own small cudaMallocs. Cap the engine
    // at 8 GiB so a couple of GiB stay outside it for the harness.
    static constexpr std::size_t device_cap_bytes = std::size_t{8} << 30;

    nntile::Context context;

    Session():
        context(
            0,
            1,
            0,
            "/tmp/nntile_ooc",
            16777216,
            0,
            "localhost",
            5001,
            0,
            host_cap_bytes,
            device_cap_bytes)
    {
        cuda_ok(cudaSetDevice(0));
    }
};

template<typename T>
void store(::nnhaul::Handle &handle, std::vector<T> const &src)
{
    auto held = handle.acquire(::nnhaul::Access::Write);
    T *dst = static_cast<T *>(held.get());
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        dst[i] = src[i];
    }
}

// The plan engine has exactly ONE chunk geometry (the one chunk class
// rule): ceil(nbytes / class) whole chunks of the one class. A legacy
// declared chunk size is ignored by Handle construction, so the
// sweep has a single variant per size (0 selects the class view).
inline std::vector<std::size_t> chunk_variants(std::size_t nbytes)
{
    (void) nbytes;
    return {std::size_t{0}};
}

// A raw chunked handle that announces itself to the nntile staging
// fallback: the nnhaul Handle API has no submit-time chunk query, and
// handles built outside core::Tile::alloc_handle are invisible to the
// stage helper's policy recompute, so the chunked legs of the wrapper
// tests must register them for staged_submit to see the chunking.
// (The rounded-chunk test mirrors the Handle constructor's effective
// decision: a single chunk covers nbytes.)
struct Chunked
{
    ::nnhaul::Handle handle;

    explicit Chunked(std::size_t nbytes, std::size_t chunk_bytes):
        handle(nbytes, chunk_bytes)
    {
        (void) chunk_bytes; // ignored: the one-class rule.
        if (nbytes > ::nnhaul::chunk_class_bytes())
        {
            // Announce the class view so the submit-time geometry
            // guards see the real launch-count arithmetic.
            ::nntile::nnhaul::stage::note_chunked(
                handle, ::nnhaul::chunk_class_bytes());
        }
    }

    ~Chunked()
    {
        // Forget before the Handle member dies so a recycled stack
        // address cannot alias a stale registry entry.
        ::nntile::nnhaul::stage::forget_handle(handle);
    }

    Chunked(Chunked const &) = delete;
    Chunked &operator=(Chunked const &) = delete;
};

template<typename T>
std::vector<T> load(::nnhaul::Handle &handle, std::size_t count)
{
    ::nnhaul::wait();
    std::vector<T> out(count);
    auto held = handle.acquire(::nnhaul::Access::Read);
    T const *src = static_cast<T const *>(held.get());
    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = src[i];
    }
    return out;
}

inline void expect_bytes(
    char const *what,
    std::vector<unsigned char> const &kernel,
    std::vector<unsigned char> const &wrapped)
{
    if (kernel.size() != wrapped.size())
    {
        throw std::runtime_error(std::string("Assert failed: ") + what);
    }
    for (std::size_t i = 0; i < kernel.size(); ++i)
    {
        if (kernel[i] != wrapped[i])
        {
            std::ostringstream message;
            message << "Assert failed: " << what << " index " << i;
            throw std::runtime_error(message.str());
        }
    }
    std::cout << "OK: " << what << "\n";
}

template<typename Y>
std::string format_real(Y value)
{
    using Bits = std::conditional_t<sizeof(Y) == sizeof(float), std::uint32_t, std::uint64_t>;
    Bits bits = 0;
    std::memcpy(&bits, &value, sizeof(Y));
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<Y>::max_digits10) << value
        << " " << std::hexfloat << value << " 0x" << std::hex << bits;
    return out.str();
}

template<typename T>
void expect_equal(
    char const *what,
    std::vector<T> const &kernel,
    std::vector<T> const &wrapped)
{
    using Y = typename T::repr_t;
    if (kernel.size() != wrapped.size())
    {
        throw std::runtime_error(std::string("Assert failed: ") + what);
    }
    for (std::size_t i = 0; i < kernel.size(); ++i)
    {
        Y const left = Y(kernel[i]);
        Y const right = Y(wrapped[i]);
        if (!(left == right))
        {
            std::ostringstream message;
            message << "Assert failed: " << what << " index " << i
                    << " kernel " << format_real(left)
                    << " wrapper " << format_real(right);
            throw std::runtime_error(message.str());
        }
    }
    std::cout << "OK: " << what << "\n";
}

template<typename T>
void expect_close(
    char const *what,
    std::vector<T> const &kernel,
    std::vector<T> const &wrapped,
    int factor)
{
    using Y = typename T::repr_t;
    if (kernel.size() != wrapped.size())
    {
        throw std::runtime_error(std::string("Assert failed: ") + what);
    }
    for (std::size_t i = 0; i < kernel.size(); ++i)
    {
        Y const left = Y(kernel[i]);
        Y const right = Y(wrapped[i]);
        Y diff = left - right;
        if (diff < Y{0})
        {
            diff = -diff;
        }
        Y scale = left;
        if (scale < Y{0})
        {
            scale = -scale;
        }
        Y const tol = static_cast<Y>(factor) * T::epsilon * scale;
        if (!(diff <= tol))
        {
            std::ostringstream message;
            message << "Assert failed: " << what << " index " << i
                    << " kernel " << format_real(left)
                    << " wrapper " << format_real(right);
            throw std::runtime_error(message.str());
        }
    }
    std::cout << "OK: " << what << "\n";
}

struct Stream
{
    cudaStream_t stream = nullptr;

    Stream()
    {
        cuda_ok(cudaStreamCreate(&stream));
    }

    ~Stream()
    {
        cudaStreamDestroy(stream);
    }

    Stream(Stream const &) = delete;
    Stream &operator=(Stream const &) = delete;
};

// multiply's partial CUDA launch writes a whole 1024-wide block. The
// direct allocation covers that block. The compare still uses the
// logical element count.
template<typename T>
struct Device
{
    T *ptr = nullptr;

    explicit Device(std::vector<T> const &host)
    {
        std::size_t const count = host.size();
        std::size_t blocks = (count + 1023) / 1024;
        if (blocks < 1)
        {
            blocks = 1;
        }
        std::size_t const alloc = blocks * 1024;
        cuda_ok(cudaMalloc(&ptr, sizeof(T) * alloc));
        cuda_ok(cudaMemset(ptr, 0, sizeof(T) * alloc));
        // join the plant (a deferred memset could zero the data
        // AFTER the following fill under device contention)
        cuda_ok(cudaStreamSynchronize(0));
        if (count > 0)
        {
            cuda_ok(cudaMemcpy(
                ptr,
                host.data(),
                sizeof(T) * count,
                cudaMemcpyHostToDevice));
        }
        cuda_ok(cudaDeviceSynchronize());
    }

    ~Device()
    {
        if (ptr != nullptr)
        {
            cudaFree(ptr);
        }
    }

    Device(Device const &) = delete;
    Device &operator=(Device const &) = delete;
};

template<typename T>
std::vector<T> from_device(T *ptr, std::size_t count, cudaStream_t stream)
{
    cuda_ok(cudaStreamSynchronize(stream));
    std::vector<T> out(count);
    if (count > 0)
    {
        cuda_ok(cudaMemcpy(
            out.data(),
            ptr,
            sizeof(T) * count,
            cudaMemcpyDeviceToHost));
    }
    return out;
}

template<typename T>
std::vector<T> series(std::size_t count, int origin, int step)
{
    using Y = typename T::repr_t;
    std::vector<T> out(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = Y(origin + step * static_cast<int>(i));
    }
    return out;
}

template<typename T>
std::vector<T> series_div(std::size_t count, int origin, int step, int denom)
{
    using Y = typename T::repr_t;
    std::vector<T> out(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = Y(origin + step * static_cast<int>(i)) / Y(denom);
    }
    return out;
}

} // namespace wrap
