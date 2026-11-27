/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/mask_scalar.cc
 * mask_scalar: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 1.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/mask_scalar.hh>
#include <nntile/nnhaul/ops/mask_scalar.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index nrows, Index ncols)
{
    using Y = typename T::repr_t;
    Scalar const value = -0.5;
    Index const nelems = nrows * ncols;
    std::vector<T> data(static_cast<std::size_t>(nelems));
    for (Index i = 0; i < nelems; ++i)
    {
        data[static_cast<std::size_t>(i)] = Y(i + 1);
    }
    std::vector<nntile::bool_t> mask(static_cast<std::size_t>(nrows));
    for (Index i = 0; i < nrows; ++i)
    {
        mask[static_cast<std::size_t>(i)] = nntile::bool_t(i % 2 != 0);
    }
    auto cpu_data = data;
    std::cout << "Run kernel::mask_scalar::cpu\n";
    nntile::kernel::mask_scalar::cpu<T>(
        nrows, ncols, mask.data(), value, cpu_data.data());
    {
        ::nnhaul::Handle mask_h(sizeof(nntile::bool_t) * mask.size());
        ::nnhaul::Handle data_h(sizeof(T) * data.size());
        wrap::store(mask_h, mask);
        wrap::store(data_h, data);
        nntile::nnhaul::mask_scalar.submit<std::tuple<T>>(
            wrap::cpu_worker, nrows, ncols, mask_h, value, data_h);
        wrap::expect_equal(
            "mask_scalar cpu", cpu_data, wrap::load<T>(data_h, data.size()));
    }
    wrap::Stream stream;
    wrap::Device<nntile::bool_t> dev_mask(mask);
    wrap::Device<T> dev_data(data);
    std::cout << "Run kernel::mask_scalar::cuda\n";
    nntile::kernel::mask_scalar::cuda<T>(
        stream.stream, nrows, ncols, dev_mask.ptr, value, dev_data.ptr);
    auto gpu_data = wrap::from_device(dev_data.ptr, data.size(), stream.stream);
    {
        ::nnhaul::Handle mask_h(sizeof(nntile::bool_t) * mask.size());
        ::nnhaul::Handle data_h(sizeof(T) * data.size());
        wrap::store(mask_h, mask);
        wrap::store(data_h, data);
        nntile::nnhaul::mask_scalar.submit<std::tuple<T>>(
            wrap::cuda_worker, nrows, ncols, mask_h, value, data_h);
        wrap::expect_equal(
            "mask_scalar cuda", gpu_data, wrap::load<T>(data_h, data.size()));
    }
    // Chunked-handle variants (the NATIVE chunked column driver: runs
    // of whole nrows-columns inside single chunks, straddling columns
    // staged through a bounded temp; the bool mask is small enough to
    // stay unchunked), same references - the piecewise result is bitwise
    // the plain kernel's.
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * data.size()))
    {
        {
            wrap::Chunked mask_h(
                sizeof(nntile::bool_t) * mask.size(), chunk);
            wrap::Chunked data_h(sizeof(T) * data.size(), chunk);
            wrap::store(mask_h.handle, mask);
            wrap::store(data_h.handle, data);
            nntile::nnhaul::mask_scalar.submit<std::tuple<T>>(
                wrap::cpu_worker, nrows, ncols, mask_h.handle, value,
                data_h.handle);
            wrap::expect_equal(
                "mask_scalar cpu chunked", cpu_data,
                wrap::load<T>(data_h.handle, data.size()));
        }
        {
            wrap::Chunked mask_h(
                sizeof(nntile::bool_t) * mask.size(), chunk);
            wrap::Chunked data_h(sizeof(T) * data.size(), chunk);
            wrap::store(mask_h.handle, mask);
            wrap::store(data_h.handle, data);
            nntile::nnhaul::mask_scalar.submit<std::tuple<T>>(
                wrap::cuda_worker, nrows, ncols, mask_h.handle, value,
                data_h.handle);
            wrap::expect_equal(
                "mask_scalar cuda chunked", gpu_data,
                wrap::load<T>(data_h.handle, data.size()));
        }
    }
}

template<typename T>
void check_straddle(Index nrows, Index ncols)
{
    // 13001-element columns over a 1.9 MB tensor: at the small-class
    // leg (1 MiB) the tensor is two chunks and a few column boundaries
    // cross the class edge, exercising the staged single-column
    // (nrows, 1) sub-call. The declared 12288 is ignored: the engine
    // has one runtime class. The 13001-byte mask may be one or several
    // chunks depending on that class, and either path is exercised.
    using Y = typename T::repr_t;
    Scalar const value = -0.75;
    Index const nelems = nrows * ncols;
    std::vector<T> data(static_cast<std::size_t>(nelems));
    for (Index i = 0; i < nelems; ++i)
    {
        data[static_cast<std::size_t>(i)] = Y(i + 1);
    }
    std::vector<nntile::bool_t> mask(static_cast<std::size_t>(nrows));
    for (Index i = 0; i < nrows; ++i)
    {
        mask[static_cast<std::size_t>(i)] =
            nntile::bool_t((i / 7) % 3 != 0);
    }
    auto cpu_data = data;
    nntile::kernel::mask_scalar::cpu<T>(nrows, ncols, mask.data(), value,
        cpu_data.data());
    {
        wrap::Chunked mask_h(sizeof(nntile::bool_t) * mask.size(),
            std::size_t{12288});
        wrap::Chunked data_h(sizeof(T) * data.size(), std::size_t{12288});
        wrap::store(mask_h.handle, mask);
        wrap::store(data_h.handle, data);
        nntile::nnhaul::mask_scalar.submit<std::tuple<T>>(
            wrap::cuda_worker, nrows, ncols, mask_h.handle, value,
            data_h.handle);
        wrap::expect_equal("mask_scalar straddle chunked", cpu_data,
            wrap::load<T>(data_h.handle, data.size()));
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    check<fp32_t>(10, 1);
    check<fp32_t>(81, 3);
    check<fp32_t>(1, 2);
    check<fp64_t>(10, 1);
    check<fp64_t>(81, 3);
    check<fp64_t>(1, 2);
    check<fp32_t>(100, 1000);
    // 13001-column boundary case over a multi-chunk tensor (fp32:
    // 52004-byte columns, fp64: 104008-byte columns).
    check_straddle<fp32_t>(13001, 37);
    check_straddle<fp64_t>(13001, 37);
    return 0;
}
