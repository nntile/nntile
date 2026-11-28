/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/subcopy.cc
 * subcopy: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 2.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/subcopy.hh>
#include <nntile/nnhaul/ops/subcopy.hh>

#include <array>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T, std::size_t NDIM>
void check(
    std::array<Index, NDIM> src_off,
    std::array<Index, NDIM> dst_off,
    std::array<Index, NDIM> shape)
{
    using Y = typename T::repr_t;
    std::vector<Index> src_start(NDIM), dst_start(NDIM), copy_shape(NDIM);
    std::vector<Index> src_shape(NDIM), dst_shape(NDIM);
    Index src_nelems = 1;
    Index dst_nelems = 1;
    for (Index i = 0; i < static_cast<Index>(NDIM); ++i)
    {
        std::size_t const at = static_cast<std::size_t>(i);
        if (src_off[at] >= 0)
        {
            src_start[at] = src_off[at];
            src_shape[at] = shape[at] + src_off[at];
        }
        else
        {
            src_start[at] = 0;
            src_shape[at] = shape[at] - src_off[at] - 1;
        }
        src_nelems *= src_shape[at];
        if (dst_off[at] >= 0)
        {
            dst_start[at] = dst_off[at];
            dst_shape[at] = shape[at] + dst_off[at];
        }
        else
        {
            dst_start[at] = 0;
            dst_shape[at] = shape[at] - dst_off[at] - 1;
        }
        dst_nelems *= dst_shape[at];
        copy_shape[at] = shape[at];
    }
    std::vector<Index> src_stride(NDIM), dst_stride(NDIM);
    src_stride[0] = 1;
    dst_stride[0] = 1;
    for (Index i = 1; i < static_cast<Index>(NDIM); ++i)
    {
        src_stride[static_cast<std::size_t>(i)] =
            src_stride[static_cast<std::size_t>(i - 1)] *
            src_shape[static_cast<std::size_t>(i - 1)];
        dst_stride[static_cast<std::size_t>(i)] =
            dst_stride[static_cast<std::size_t>(i - 1)] *
            dst_shape[static_cast<std::size_t>(i - 1)];
    }
    std::vector<T> src(static_cast<std::size_t>(src_nelems));
    std::vector<T> dst(static_cast<std::size_t>(dst_nelems));
    for (Index i = 0; i < src_nelems; ++i)
    {
        src[static_cast<std::size_t>(i)] = Y(i + 1);
    }
    for (Index i = 0; i < dst_nelems; ++i)
    {
        dst[static_cast<std::size_t>(i)] = Y(-i - 1);
    }
    auto cpu_dst = dst;
    std::vector<nntile::int64_t> tmp(2 * NDIM);
    std::cout << "Run kernel::subcopy::cpu\n";
    nntile::kernel::subcopy::cpu<T>(
        static_cast<Index>(NDIM),
        src_start.data(),
        src_stride.data(),
        copy_shape.data(),
        src.data(),
        dst_start.data(),
        dst_stride.data(),
        cpu_dst.data(),
        tmp.data());
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        ::nnhaul::Handle tmp_h(sizeof(nntile::int64_t) * tmp.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        wrap::store(tmp_h, tmp);
        nntile::nnhaul::subcopy.submit<std::tuple<T>>(
            wrap::cpu_worker,
            static_cast<Index>(NDIM),
            src_start,
            src_stride,
            dst_start,
            dst_stride,
            copy_shape,
            src_h,
            dst_h,
            tmp_h,
            ACCESS_RW);
        wrap::expect_equal(
            "subcopy cpu", cpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    wrap::Stream stream;
    wrap::Device<T> dev_src(src);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::subcopy::cuda\n";
    nntile::kernel::subcopy::cuda<T>(
        stream.stream,
        static_cast<Index>(NDIM),
        src_start.data(),
        src_stride.data(),
        copy_shape.data(),
        dev_src.ptr,
        dst_start.data(),
        dst_stride.data(),
        dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        ::nnhaul::Handle tmp_h(sizeof(nntile::int64_t) * tmp.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        wrap::store(tmp_h, tmp);
        nntile::nnhaul::subcopy.submit<std::tuple<T>>(
            wrap::cuda_worker,
            static_cast<Index>(NDIM),
            src_start,
            src_stride,
            dst_start,
            dst_stride,
            copy_shape,
            src_h,
            dst_h,
            tmp_h,
            ACCESS_RW);
        wrap::expect_equal(
            "subcopy cuda", gpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    // Chunked-handle variants (staged fallback; the RW dst and the
    // int64 tmp buffer stage too), same references.
    for (std::size_t chunk :
        wrap::chunk_variants(sizeof(T) * (src_nelems > dst_nelems
                                 ? src_nelems
                                 : dst_nelems)))
    {
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::Chunked tmp_h(sizeof(nntile::int64_t) * tmp.size(), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            wrap::store(tmp_h.handle, tmp);
            nntile::nnhaul::subcopy.submit<std::tuple<T>>(
                wrap::cpu_worker,
                static_cast<Index>(NDIM),
                src_start,
                src_stride,
                dst_start,
                dst_stride,
                copy_shape,
                src_h.handle,
                dst_h.handle,
                tmp_h.handle,
                ACCESS_RW);
            wrap::expect_equal(
                "subcopy cpu chunked", cpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()));
        }
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::Chunked tmp_h(sizeof(nntile::int64_t) * tmp.size(), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            wrap::store(tmp_h.handle, tmp);
            nntile::nnhaul::subcopy.submit<std::tuple<T>>(
                wrap::cuda_worker,
                static_cast<Index>(NDIM),
                src_start,
                src_stride,
                dst_start,
                dst_stride,
                copy_shape,
                src_h.handle,
                dst_h.handle,
                tmp_h.handle,
                ACCESS_RW);
            wrap::expect_equal(
                "subcopy cuda chunked", gpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()));
        }
    }
}

template<typename T>
void check_many()
{
    check<T, 1>({0}, {0}, {2});
    check<T, 1>({2}, {0}, {2});
    check<T, 1>({-2}, {0}, {2});
    check<T, 1>({0}, {2}, {2});
    check<T, 1>({0}, {-2}, {2});
    check<T, 3>({0, 0, 0}, {0, 0, 0}, {2, 3, 4});
    check<T, 3>({1, 0, 0}, {0, 0, 0}, {2, 3, 4});
    check<T, 3>({1, 0, 0}, {-1, 0, 0}, {2, 3, 4});
    check<T, 3>({0, 1, -1}, {3, -4, 5}, {2, 3, 4});
    // Big 1D tiles (200000 elems) so both chunk variants chunk.
    check<T, 1>({0}, {0}, {200000});
    // Family-3 native-driver legs: a dense 2-D window whose stride[1]
    // equals copy_shape[0] (batchable - multi-column 2-D sub-call
    // pieces), and a straddling-column torture (3001-element columns
    // vs 3072/1536-element chunks: every column cut into partial
    // 1-D sub-pieces at the boundaries).
    check<T, 2>({0, 0}, {0, 0}, {17, 20000});
    check<T, 2>({0, 0}, {0, 0}, {3001, 41});
}

int main()
{
    wrap::Session session;
    (void)session;
    check_many<fp32_t>();
    check_many<fp64_t>();
    return 0;
}
