/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file nntile/tests/kernel/chunked_ops.cc
 * Kernel-level A/B for the chunked entry points that the small wrapper
 * tests cannot reach: every tensor here spans many chunk-class pieces
 * plus a short tail. Each op is run through its classic contiguous entry
 * point and its chunked entry point on identical bytes, and the results
 * must agree bitwise. Built only when CUDA is enabled; the kernel entry
 * points themselves are backend-agnostic.
 *
 * @version 2.0.0
 * */

#include <nntile/kernel/add_fiber/cuda.hh>
#include <nntile/kernel/clear/cuda.hh>
#include <nntile/kernel/add_fiber_inplace/cuda.hh>
#include <nntile/kernel/add_inplace/cuda.hh>
#include <nntile/kernel/add_slice/cuda.hh>
#include <nntile/kernel/add_slice_inplace/cuda.hh>
#include <nntile/kernel/fill/cuda.hh>
#include <nntile/kernel/gelu/cuda.hh>
#include <nntile/kernel/gelutanh/cuda.hh>
#include <nntile/kernel/gelutanh_backward/cuda.hh>
#include <nntile/kernel/hypot_scalar_inverse/cuda.hh>
#include <nntile/kernel/logsumexp/cuda.hh>
#include <nntile/kernel/mask_scalar/cuda.hh>
#include <nntile/kernel/maxsumexp/cuda.hh>
#include <nntile/kernel/multiply_fiber/cuda.hh>
#include <nntile/kernel/multiply_inplace/cuda.hh>
#include <nntile/kernel/multiply_slice/cuda.hh>
#include <nntile/kernel/scale/cuda.hh>
#include <nntile/kernel/sgd_step/cuda.hh>
#include <nntile/kernel/softmax/cuda.hh>
#include <nntile/kernel/subtract_indexed_outputs/cuda.hh>
#include <nntile/kernel/sum_fiber/cuda.hh>
#include <nntile/kernel/sum_slice/cuda.hh>
#include <nntile/kernel/sumprod_fiber/cuda.hh>
#include <nntile/kernel/sumprod_slice/cuda.hh>
#include <nntile/kernel/total_sum_accum/cuda.hh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using nntile::Index;
using nntile::bool_t;
using nntile::fp32_t;
using nntile::kernel::ChunkClass;

namespace
{

#define CK(x) do { cudaError_t e_ = (x); if(e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
        cudaGetErrorString(e_)); std::exit(2); } } while(0)

//! One operand: a chunk chain plus the contiguous twin the classic
//! entry point sees, both holding the same bytes.
struct Operand
{
    std::vector<void *> ptrs;
    void **table = nullptr;
    Index nchunks = 0;
    fp32_t *flat = nullptr;
    std::size_t nbytes = 0;

    void alloc(std::size_t bytes, std::size_t class_bytes)
    {
        nbytes = bytes;
        std::size_t const chunk = class_bytes;
        nchunks = Index((bytes + chunk - 1) / chunk);
        ptrs.resize(nchunks);
        for(Index i = 0; i < nchunks; ++i)
        {
            std::size_t const off = i * chunk;
            std::size_t const len = std::min(chunk, bytes - off);
            CK(cudaMalloc(&ptrs[i], len));
        }
        CK(cudaMalloc(&table, nchunks * sizeof(void *)));
        CK(cudaMemcpy(table, ptrs.data(), nchunks * sizeof(void *),
            cudaMemcpyHostToDevice));
        CK(cudaMalloc(&flat, bytes));
    }

    //! Fill both representations from one host vector.
    void upload(std::vector<fp32_t> const &src, std::size_t class_bytes)
    {
        std::size_t const chunk_elems = class_bytes / sizeof(fp32_t);
        CK(cudaMemcpy(flat, src.data(), nbytes, cudaMemcpyHostToDevice));
        for(Index c = 0; c < nchunks; ++c)
        {
            std::size_t const off = std::size_t(c) * chunk_elems;
            Index const len = std::min<Index>(Index(chunk_elems),
                Index((nbytes / sizeof(fp32_t)) - off));
            CK(cudaMemcpy(ptrs[c], src.data() + off,
                std::size_t(len) * sizeof(fp32_t), cudaMemcpyHostToDevice));
        }
    }

    std::vector<fp32_t> gather(std::size_t class_bytes) const
    {
        std::vector<fp32_t> out(nbytes / sizeof(fp32_t));
        std::size_t const chunk_elems = class_bytes / sizeof(fp32_t);
        for(Index c = 0; c < nchunks; ++c)
        {
            std::size_t const off = std::size_t(c) * chunk_elems;
            Index const len = std::min<Index>(Index(chunk_elems),
                Index(out.size() - off));
            CK(cudaMemcpy(out.data() + off, ptrs[c],
                std::size_t(len) * sizeof(fp32_t), cudaMemcpyDeviceToHost));
        }
        return out;
    }

    std::vector<fp32_t> gather_flat() const
    {
        std::vector<fp32_t> out(nbytes / sizeof(fp32_t));
        CK(cudaMemcpy(out.data(), flat, nbytes, cudaMemcpyDeviceToHost));
        return out;
    }

    ~Operand()
    {
        for(void *p : ptrs)
        {
            cudaFree(p);
        }
        cudaFree(table);
        cudaFree(flat);
    }
};

int failures = 0;

void check(char const *name, std::vector<fp32_t> const &a,
    std::vector<fp32_t> const &b)
{
    if(a.size() != b.size())
    {
        std::fprintf(stderr, "%s: size mismatch %zu vs %zu\n", name,
            a.size(), b.size());
        ++failures;
        return;
    }
    for(std::size_t i = 0; i < a.size(); ++i)
    {
        if(a[i].value != b[i].value)
        {
            std::fprintf(stderr, "%s: mismatch at %zu: %f vs %f\n", name, i,
                (double)a[i].value, (double)b[i].value);
            ++failures;
            return;
        }
    }
    std::printf("%s: %zu elements identical\n", name, a.size());
}

ChunkClass make_class(std::size_t bytes)
{
    ChunkClass cls;
    cls.mask = bytes - 1;
    cls.shift = 0;
    for(std::size_t v = bytes; (v >>= 1) != 0;)
    {
        ++cls.shift;
    }
    return cls;
}

// 64 KiB class: a 4 MiB tensor is 64 pieces, far past any single-chunk
// special case, and the odd tail is not a class multiple.
constexpr std::size_t kClassBytes = 64u << 10;

std::vector<fp32_t> pattern(std::size_t n, float scale, float bias,
    unsigned mod)
{
    std::vector<fp32_t> out(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        out[i] = fp32_t(scale * float(i % mod) + bias);
    }
    return out;
}

} // namespace

int main()
{
    ChunkClass const cls = make_class(kClassBytes);
    Index const chunk_elems = Index(kClassBytes / sizeof(fp32_t));
    // Single-chunk operands carry their real element count, as the
    // production ChunkView does; only a multi-chunk operand uses the class.
    auto elems = [chunk_elems](Operand const &o) {
        return o.nchunks == 1
            ? Index(o.nbytes / sizeof(fp32_t)) : chunk_elems;
    };
    // 10 pieces and a 3-element tail.
    Index const nelems = 10 * chunk_elems + 3;
    std::size_t const nbytes = std::size_t(nelems) * sizeof(fp32_t);

    cudaStream_t s;
    CK(cudaStreamCreate(&s));


    // ---- gelu: unary, out of place ----
    {
        std::vector<fp32_t> h = pattern(nelems, 0.002f, -5.0f, 991);
        Operand src, dst;
        src.alloc(nbytes, kClassBytes);
        dst.alloc(nbytes, kClassBytes);
        src.upload(h, kClassBytes);
        nntile::kernel::gelu::cuda<fp32_t>(s, nelems, src.flat, dst.flat);
        nntile::kernel::gelu::chunked<fp32_t>(s, nelems,
            (const fp32_t * const *)src.table, src.nchunks,
            (fp32_t * const *)dst.table, dst.nchunks,
            chunk_elems, chunk_elems, cls);
        CK(cudaStreamSynchronize(s));
        check("gelu", dst.gather_flat(), dst.gather(kClassBytes));
    }

    // ---- gelutanh: unary, out of place ----
    {
        std::vector<fp32_t> h = pattern(nelems, 0.003f, -4.0f, 2711);
        Operand src, dst;
        src.alloc(nbytes, kClassBytes);
        dst.alloc(nbytes, kClassBytes);
        src.upload(h, kClassBytes);
        nntile::kernel::gelutanh::cuda<fp32_t>(s, nelems, src.flat, dst.flat);
        nntile::kernel::gelutanh::chunked<fp32_t>(s, nelems,
            (const fp32_t * const *)src.table, src.nchunks,
            (fp32_t * const *)dst.table, dst.nchunks,
            chunk_elems, chunk_elems, cls);
        CK(cudaStreamSynchronize(s));
        check("gelutanh", dst.gather_flat(), dst.gather(kClassBytes));
    }

    // ---- scale: unary with a factor, out of place ----
    {
        std::vector<fp32_t> h = pattern(nelems, 0.005f, -7.0f, 1237);
        Operand src, dst;
        src.alloc(nbytes, kClassBytes);
        dst.alloc(nbytes, kClassBytes);
        src.upload(h, kClassBytes);
        nntile::kernel::scale::cuda<fp32_t>(s, nelems, -1.75, src.flat,
            dst.flat);
        nntile::kernel::scale::chunked<fp32_t>(s, nelems, -1.75,
            (const fp32_t * const *)src.table, src.nchunks,
            (fp32_t * const *)dst.table, dst.nchunks,
            chunk_elems, chunk_elems, cls);
        CK(cudaStreamSynchronize(s));
        check("scale", dst.gather_flat(), dst.gather(kClassBytes));
    }

    // ---- add_inplace: dst accumulates src, both multi-chunk ----
    {
        std::vector<fp32_t> a = pattern(nelems, 0.002f, -3.0f, 977);
        std::vector<fp32_t> b = pattern(nelems, 0.004f, 1.5f, 613);
        Operand src_c, dst_c, src_h, dst_h;
        src_c.alloc(nbytes, kClassBytes);
        dst_c.alloc(nbytes, kClassBytes);
        src_h.alloc(nbytes, kClassBytes);
        dst_h.alloc(nbytes, kClassBytes);
        src_c.upload(a, kClassBytes);
        dst_c.upload(b, kClassBytes);
        src_h.upload(a, kClassBytes);
        dst_h.upload(b, kClassBytes);
        nntile::kernel::add_inplace::cuda<fp32_t>(s, nelems, 0.75, src_h.flat,
            2.5, dst_h.flat);
        nntile::kernel::add_inplace::chunked<fp32_t>(s, nelems, 0.75,
            (const fp32_t * const *)src_c.table, src_c.nchunks, 2.5,
            (fp32_t * const *)dst_c.table, dst_c.nchunks,
            chunk_elems, chunk_elems, cls);
        CK(cudaStreamSynchronize(s));
        check("add_inplace", dst_h.gather_flat(), dst_c.gather(kClassBytes));
    }

    // ---- fill: write-only, multi-chunk destination ----
    {
        Operand dst_c, dst_h;
        dst_c.alloc(nbytes, kClassBytes);
        dst_h.alloc(nbytes, kClassBytes);
        nntile::kernel::fill::cuda<fp32_t>(s, nelems, 0.375, dst_h.flat);
        nntile::kernel::fill::chunked<fp32_t>(s, nelems, 0.375,
            (fp32_t * const *)dst_c.table, dst_c.nchunks, chunk_elems, cls);
        CK(cudaStreamSynchronize(s));
        check("fill", dst_h.gather_flat(), dst_c.gather(kClassBytes));
    }

    // ---- sumprod_fiber: cross-chunk reduction over n ----
    {
        Index const m = 1, n = nelems, k = 1;
        std::size_t const src_bytes = std::size_t(m * n * k) * sizeof(fp32_t);
        std::size_t const dst_bytes = std::size_t(k) * sizeof(fp32_t);
        Operand s1c, s2c, dc, s1h, s2h, dh;
        s1c.alloc(src_bytes, kClassBytes);
        s2c.alloc(src_bytes, kClassBytes);
        dc.alloc(dst_bytes, kClassBytes);
        s1h.alloc(src_bytes, kClassBytes);
        s2h.alloc(src_bytes, kClassBytes);
        dh.alloc(dst_bytes, kClassBytes);
        std::vector<fp32_t> a = pattern(std::size_t(m * n * k), 0.001f,
            -2.0f, 541);
        std::vector<fp32_t> b = pattern(std::size_t(m * n * k), 0.002f,
            0.5f, 829);
        std::vector<fp32_t> z(std::size_t(k), fp32_t(1.0f));
        s1c.upload(a, kClassBytes);
        s2c.upload(b, kClassBytes);
        dc.upload(z, kClassBytes);
        s1h.upload(a, kClassBytes);
        s2h.upload(b, kClassBytes);
        dh.upload(z, kClassBytes);
        nntile::kernel::sumprod_fiber::cuda<fp32_t>(s, m, n, k, 0.5,
            s1h.flat, s2h.flat, 3.0, dh.flat);
        nntile::kernel::sumprod_fiber::chunked<fp32_t>(s, m, n, k, 0.5,
            (const fp32_t * const *)s1c.table, s1c.nchunks,
            (const fp32_t * const *)s2c.table, s2c.nchunks, 3.0,
            (fp32_t * const *)dc.table, dc.nchunks,
            chunk_elems, chunk_elems, elems(dc), cls);
        CK(cudaStreamSynchronize(s));
        check("sumprod_fiber", dh.gather_flat(), dc.gather(kClassBytes));
    }

    // ---- sum_slice: reduction along k, fibers span many chunks ----
    {
        Index const m = 1, n = 2, k = nelems;
        std::size_t const src_bytes = std::size_t(m * n * k) * sizeof(fp32_t);
        std::size_t const dst_bytes = std::size_t(m * n) * sizeof(fp32_t);
        Operand sc, dh, sh, dc;
        sc.alloc(src_bytes, kClassBytes);
        sh.alloc(src_bytes, kClassBytes);
        dc.alloc(dst_bytes, kClassBytes);
        dh.alloc(dst_bytes, kClassBytes);
        std::vector<fp32_t> a = pattern(std::size_t(m * n * k), 0.002f,
            -1.0f, 359);
        std::vector<fp32_t> z(std::size_t(m * n), fp32_t(0.5f));
        sc.upload(a, kClassBytes);
        sh.upload(a, kClassBytes);
        dc.upload(z, kClassBytes);
        dh.upload(z, kClassBytes);
        nntile::kernel::sum_slice::cuda<fp32_t>(s, m, n, k, 0.25, sh.flat,
            1.5, dh.flat);
        nntile::kernel::sum_slice::chunked<fp32_t>(s, m, n, k, 0.25,
            (const fp32_t * const *)sc.table, sc.nchunks, 1.5,
            (fp32_t * const *)dc.table, dc.nchunks,
            chunk_elems, elems(dc), cls);
        CK(cudaStreamSynchronize(s));
        check("sum_slice", dh.gather_flat(), dc.gather(kClassBytes));
    }

    // ---- maxsumexp: per-fiber max and sum over a long k axis ----
    {
        Index const m = 1, n = 2, k = nelems;
        std::size_t const src_bytes = std::size_t(m * n * k) * sizeof(fp32_t);
        std::size_t const dst_bytes = std::size_t(2 * m * n) * sizeof(fp32_t);
        Operand sc, sh, dc, dh;
        sc.alloc(src_bytes, kClassBytes);
        sh.alloc(src_bytes, kClassBytes);
        dc.alloc(dst_bytes, kClassBytes);
        dh.alloc(dst_bytes, kClassBytes);
        std::vector<fp32_t> a = pattern(std::size_t(m * n * k), 0.004f,
            -5.0f, 1451);
        std::vector<fp32_t> z(std::size_t(2 * m * n), fp32_t(0.0f));
        sc.upload(a, kClassBytes);
        sh.upload(a, kClassBytes);
        dc.upload(z, kClassBytes);
        dh.upload(z, kClassBytes);
        nntile::kernel::maxsumexp::cuda<fp32_t>(s, m, n, k, sh.flat, 0.0,
            dh.flat);
        nntile::kernel::maxsumexp::chunked<fp32_t>(s, m, n, k,
            (const fp32_t * const *)sc.table, sc.nchunks, 0.0,
            (fp32_t * const *)dc.table, dc.nchunks, chunk_elems,
            elems(dc), long(m * n * k), long(2 * m * n), cls);
        CK(cudaStreamSynchronize(s));
        check("maxsumexp", dh.gather_flat(), dc.gather(kClassBytes));
    }

    // ---- softmax: per-fiber normalization over a long k axis ----
    {
        Index const m = 1, n = 2, k = nelems;
        std::size_t const src_bytes = std::size_t(m * n * k) * sizeof(fp32_t);
        std::size_t const mse_bytes = std::size_t(2 * m * n) * sizeof(fp32_t);
        Operand sc, sh, mc, mh, dc, dh;
        sc.alloc(src_bytes, kClassBytes);
        sh.alloc(src_bytes, kClassBytes);
        dc.alloc(src_bytes, kClassBytes);
        dh.alloc(src_bytes, kClassBytes);
        mc.alloc(mse_bytes, kClassBytes);
        mh.alloc(mse_bytes, kClassBytes);
        std::vector<fp32_t> a = pattern(std::size_t(m * n * k), 0.003f,
            -3.5f, 1709);
        std::vector<fp32_t> mse(std::size_t(2 * m * n));
        for(Index j = 0; j < m * n; ++j)
        {
            mse[std::size_t(2 * j)] = fp32_t(0.125f);
            mse[std::size_t(2 * j + 1)] = fp32_t(37.5f);
        }
        sc.upload(a, kClassBytes);
        sh.upload(a, kClassBytes);
        dc.upload(a, kClassBytes);
        dh.upload(a, kClassBytes);
        mc.upload(mse, kClassBytes);
        mh.upload(mse, kClassBytes);
        nntile::kernel::softmax::cuda<fp32_t>(s, m, n, k, mh.flat, sh.flat,
            1.25, dh.flat);
        nntile::kernel::softmax::chunked<fp32_t>(s, m, n, k,
            (const fp32_t * const *)mc.table, mc.nchunks,
            (const fp32_t * const *)sc.table, sc.nchunks, 1.25,
            (fp32_t * const *)dc.table, dc.nchunks,
            elems(mc), chunk_elems, chunk_elems, cls);
        CK(cudaStreamSynchronize(s));
        check("softmax", dh.gather_flat(), dc.gather(kClassBytes));
    }

    // ---- multiply_inplace: dst[i] = alpha*src[i]*dst[i], multi-chunk ----
    {
        std::vector<fp32_t> a = pattern(nelems, 0.002f, -3.0f, 977);
        std::vector<fp32_t> b = pattern(nelems, 0.004f, 1.5f, 613);
        Operand src_c, dst_c, src_h, dst_h;
        src_c.alloc(nbytes, kClassBytes);
        dst_c.alloc(nbytes, kClassBytes);
        src_h.alloc(nbytes, kClassBytes);
        dst_h.alloc(nbytes, kClassBytes);
        src_c.upload(a, kClassBytes);
        dst_c.upload(b, kClassBytes);
        src_h.upload(a, kClassBytes);
        dst_h.upload(b, kClassBytes);
        nntile::kernel::multiply_inplace::cuda<fp32_t>(s, nelems, -0.5,
            src_h.flat, dst_h.flat);
        nntile::kernel::multiply_inplace::chunked<fp32_t>(s, nelems, -0.5,
            (const fp32_t * const *)src_c.table, src_c.nchunks,
            chunk_elems, (fp32_t * const *)dst_c.table, dst_c.nchunks,
            chunk_elems, cls);
        CK(cudaStreamSynchronize(s));
        check("multiply_inplace", dst_h.gather_flat(),
            dst_c.gather(kClassBytes));
    }

    // ---- hypot_scalar_inverse: write-only, multi-chunk destination ----
    {
        std::vector<fp32_t> a = pattern(nelems, 0.006f, -3.0f, 2311);
        Operand dst_c, dst_h;
        dst_c.alloc(nbytes, kClassBytes);
        dst_h.alloc(nbytes, kClassBytes);
        dst_c.upload(a, kClassBytes);
        dst_h.upload(a, kClassBytes);
        nntile::kernel::hypot_scalar_inverse::cuda<fp32_t>(s, nelems,
            0.125, -2.0, dst_h.flat);
        nntile::kernel::hypot_scalar_inverse::chunked<fp32_t>(s, nelems,
            0.125, -2.0,
            (fp32_t * const *)dst_c.table, dst_c.nchunks, chunk_elems,
            cls);
        CK(cudaStreamSynchronize(s));
        check("hypot_scalar_inverse", dst_h.gather_flat(),
            dst_c.gather(kClassBytes));
    }

    // ---- gelutanh_backward: two inputs and an accumulate operand ----
    {
        std::vector<fp32_t> x = pattern(nelems, 0.003f, -4.0f, 2711);
        std::vector<fp32_t> dy = pattern(nelems, 0.002f, 0.5f, 401);
        std::vector<fp32_t> dx0 = pattern(nelems, 0.001f, -1.0f, 89);
        Operand xc, yc, dc, xh, yh, dh;
        xc.alloc(nbytes, kClassBytes);
        yc.alloc(nbytes, kClassBytes);
        dc.alloc(nbytes, kClassBytes);
        xh.alloc(nbytes, kClassBytes);
        yh.alloc(nbytes, kClassBytes);
        dh.alloc(nbytes, kClassBytes);
        xc.upload(x, kClassBytes);
        yc.upload(dy, kClassBytes);
        dc.upload(dx0, kClassBytes);
        xh.upload(x, kClassBytes);
        yh.upload(dy, kClassBytes);
        dh.upload(dx0, kClassBytes);
        nntile::kernel::gelutanh_backward::cuda<fp32_t>(s, nelems, 1.25,
            xh.flat, yh.flat, 0.75, dh.flat);
        nntile::kernel::gelutanh_backward::chunked<fp32_t>(s, nelems, 1.25,
            (const fp32_t * const *)xc.table, xc.nchunks,
            (const fp32_t * const *)yc.table, yc.nchunks, 0.75,
            (fp32_t * const *)dc.table, dc.nchunks,
            chunk_elems, chunk_elems, chunk_elems, cls);
        CK(cudaStreamSynchronize(s));
        check("gelutanh_backward", dh.gather_flat(),
            dc.gather(kClassBytes));
    }

    // ---- logsumexp: 2*nelems input to nelems output, both multi-chunk ----
    {
        Index const nmse = nelems;
        std::size_t const mse_bytes = std::size_t(2 * nmse) * sizeof(fp32_t);
        std::size_t const lse_bytes = std::size_t(nmse) * sizeof(fp32_t);
        Operand mc, mh, dc, dh;
        mc.alloc(mse_bytes, kClassBytes);
        mh.alloc(mse_bytes, kClassBytes);
        dc.alloc(lse_bytes, kClassBytes);
        dh.alloc(lse_bytes, kClassBytes);
        std::vector<fp32_t> mse(std::size_t(2 * nmse));
        for(Index j = 0; j < nmse; ++j)
        {
            mse[std::size_t(2 * j)] = fp32_t(0.25f);
            mse[std::size_t(2 * j + 1)] = fp32_t(0.5f + float(j % 97));
        }
        std::vector<fp32_t> lse(std::size_t(nmse), fp32_t(0.0f));
        mc.upload(mse, kClassBytes);
        mh.upload(mse, kClassBytes);
        dc.upload(lse, kClassBytes);
        dh.upload(lse, kClassBytes);
        nntile::kernel::logsumexp::cuda<fp32_t>(s, nmse, mh.flat, dh.flat);
        nntile::kernel::logsumexp::chunked<fp32_t>(s, nmse,
            (const fp32_t * const *)mc.table, mc.nchunks,
            (fp32_t * const *)dc.table, dc.nchunks, chunk_elems,
            chunk_elems, cls);
        CK(cudaStreamSynchronize(s));
        check("logsumexp", dh.gather_flat(), dc.gather(kClassBytes));
    }

    // ---- add_slice: tensor + broadcast slice, all multi-chunk ----
    {
        Index const m = 1, n = nelems, k = 1;
        Operand a_c, b_c, d_c, a_h, b_h, d_h;
        a_c.alloc(nbytes, kClassBytes);
        b_c.alloc(nbytes, kClassBytes);
        d_c.alloc(nbytes, kClassBytes);
        a_h.alloc(nbytes, kClassBytes);
        b_h.alloc(nbytes, kClassBytes);
        d_h.alloc(nbytes, kClassBytes);
        std::vector<fp32_t> a = pattern(nelems, 0.002f, -1.0f, 313);
        std::vector<fp32_t> b = pattern(nelems, 0.003f, 2.0f, 719);
        std::vector<fp32_t> d = pattern(nelems, 0.001f, -0.5f, 127);
        a_c.upload(a, kClassBytes);
        b_c.upload(b, kClassBytes);
        d_c.upload(d, kClassBytes);
        a_h.upload(a, kClassBytes);
        b_h.upload(b, kClassBytes);
        d_h.upload(d, kClassBytes);
        nntile::kernel::add_slice::cuda<fp32_t>(s, m, n, k, 1.25, a_h.flat,
            -0.75, b_h.flat, d_h.flat);
        nntile::kernel::add_slice::chunked<fp32_t>(s, m, n, k, 1.25,
            (const fp32_t * const *)a_c.table, a_c.nchunks, -0.75,
            (const fp32_t * const *)b_c.table, b_c.nchunks,
            (fp32_t * const *)d_c.table, d_c.nchunks,
            chunk_elems, chunk_elems, chunk_elems, cls);
        CK(cudaStreamSynchronize(s));
        check("add_slice", d_h.gather_flat(), d_c.gather(kClassBytes));
    }

    // ---- multiply_slice: broadcast product into the tensor ----
    {
        Index const m = 1, n = nelems, k = 1;
        Operand s_c, d_c, s_h, d_h;
        s_c.alloc(nbytes, kClassBytes);
        d_c.alloc(nbytes, kClassBytes);
        s_h.alloc(nbytes, kClassBytes);
        d_h.alloc(nbytes, kClassBytes);
        std::vector<fp32_t> a = pattern(nelems, 0.002f, -1.0f, 313);
        std::vector<fp32_t> d = pattern(nelems, 0.004f, 1.5f, 823);
        s_c.upload(a, kClassBytes);
        d_c.upload(d, kClassBytes);
        s_h.upload(a, kClassBytes);
        d_h.upload(d, kClassBytes);
        nntile::kernel::multiply_slice::cuda<fp32_t>(s, m, n, k, -0.5,
            s_h.flat, d_h.flat);
        nntile::kernel::multiply_slice::chunked<fp32_t>(s, m, n, k, -0.5,
            (const fp32_t * const *)s_c.table, s_c.nchunks, chunk_elems,
            (fp32_t * const *)d_c.table, d_c.nchunks, chunk_elems, cls);
        CK(cudaStreamSynchronize(s));
        check("multiply_slice", d_h.gather_flat(), d_c.gather(kClassBytes));
    }

    // ---- add_slice_inplace: broadcast slice accumulated into dst ----
    {
        Index const m = 1, n = nelems, k = 1;
        Operand s_c, d_c, s_h, d_h;
        s_c.alloc(nbytes, kClassBytes);
        d_c.alloc(nbytes, kClassBytes);
        s_h.alloc(nbytes, kClassBytes);
        d_h.alloc(nbytes, kClassBytes);
        std::vector<fp32_t> a = pattern(nelems, 0.002f, -1.0f, 313);
        std::vector<fp32_t> d = pattern(nelems, 0.004f, 1.5f, 823);
        s_c.upload(a, kClassBytes);
        d_c.upload(d, kClassBytes);
        s_h.upload(a, kClassBytes);
        d_h.upload(d, kClassBytes);
        nntile::kernel::add_slice_inplace::cuda<fp32_t>(s, m, n, k, 0.75,
            s_h.flat, 2.0, d_h.flat);
        nntile::kernel::add_slice_inplace::chunked<fp32_t>(s, m, n, k,
            0.75, (const fp32_t * const *)s_c.table, s_c.nchunks, 2.0,
            (fp32_t * const *)d_c.table, d_c.nchunks, chunk_elems,
            chunk_elems, cls);
        CK(cudaStreamSynchronize(s));
        check("add_slice_inplace", d_h.gather_flat(),
            d_c.gather(kClassBytes));
    }

    // ---- sumprod_slice: product reduction along a long k axis ----
    {
        Index const m = 1, n = 2, k = nelems;
        std::size_t const src_bytes = std::size_t(m * n * k) * sizeof(fp32_t);
        std::size_t const dst_bytes = std::size_t(m * n) * sizeof(fp32_t);
        Operand s1c, s2c, dc, s1h, s2h, dh;
        s1c.alloc(src_bytes, kClassBytes);
        s2c.alloc(src_bytes, kClassBytes);
        dc.alloc(dst_bytes, kClassBytes);
        s1h.alloc(src_bytes, kClassBytes);
        s2h.alloc(src_bytes, kClassBytes);
        dh.alloc(dst_bytes, kClassBytes);
        std::vector<fp32_t> a = pattern(std::size_t(m * n * k), 0.001f,
            -2.0f, 541);
        std::vector<fp32_t> b = pattern(std::size_t(m * n * k), 0.002f,
            0.5f, 829);
        std::vector<fp32_t> z(std::size_t(m * n), fp32_t(0.5f));
        s1c.upload(a, kClassBytes);
        s2c.upload(b, kClassBytes);
        dc.upload(z, kClassBytes);
        s1h.upload(a, kClassBytes);
        s2h.upload(b, kClassBytes);
        dh.upload(z, kClassBytes);
        nntile::kernel::sumprod_slice::cuda<fp32_t>(s, m, n, k, 0.5,
            s1h.flat, s2h.flat, 1.5, dh.flat);
        nntile::kernel::sumprod_slice::chunked<fp32_t>(s, m, n, k, 0.5,
            (const fp32_t * const *)s1c.table, s1c.nchunks,
            (const fp32_t * const *)s2c.table, s2c.nchunks, 1.5,
            (fp32_t * const *)dc.table, dc.nchunks,
            chunk_elems, chunk_elems, elems(dc), cls);
        CK(cudaStreamSynchronize(s));
        check("sumprod_slice", dh.gather_flat(), dc.gather(kClassBytes));
    }

    // ---- add_fiber: tensor plus broadcast fiber ----
    {
        Index const m = 1, n = 1, k = nelems, batch = 1;
        std::size_t const fiber_bytes = std::size_t(k) * sizeof(fp32_t);
        Operand s1c, s2c, dc, s1h, s2h, dh;
        s1c.alloc(fiber_bytes, kClassBytes);
        s2c.alloc(nbytes, kClassBytes);
        dc.alloc(nbytes, kClassBytes);
        s1h.alloc(fiber_bytes, kClassBytes);
        s2h.alloc(nbytes, kClassBytes);
        dh.alloc(nbytes, kClassBytes);
        std::vector<fp32_t> f1 = pattern(std::size_t(k), 0.002f, -1.0f, 313);
        std::vector<fp32_t> a = pattern(nelems, 0.003f, 2.0f, 719);
        std::vector<fp32_t> d = pattern(nelems, 0.001f, -0.5f, 127);
        s1c.upload(f1, kClassBytes);
        s2c.upload(a, kClassBytes);
        dc.upload(d, kClassBytes);
        s1h.upload(f1, kClassBytes);
        s2h.upload(a, kClassBytes);
        dh.upload(d, kClassBytes);
        nntile::kernel::add_fiber::cuda<fp32_t>(s, m, n, k, batch, 1.25,
            s1h.flat, -0.75, s2h.flat, dh.flat);
        nntile::kernel::add_fiber::chunked<fp32_t>(s, m, n, k, batch, 1.25,
            (const fp32_t * const *)s1c.table, s1c.nchunks, -0.75,
            (const fp32_t * const *)s2c.table, s2c.nchunks,
            (fp32_t * const *)dc.table, dc.nchunks,
            elems(s1c), elems(s2c), elems(dc), cls);
        CK(cudaStreamSynchronize(s));
        check("add_fiber", dh.gather_flat(), dc.gather(kClassBytes));
    }

    // ---- add_fiber_inplace: broadcast fiber accumulated into dst ----
    {
        Index const m = 1, n = 1, k = nelems, batch = 1;
        std::size_t const fiber_bytes = std::size_t(k) * sizeof(fp32_t);
        Operand sc, dc, sh, dh;
        sc.alloc(fiber_bytes, kClassBytes);
        dc.alloc(nbytes, kClassBytes);
        sh.alloc(fiber_bytes, kClassBytes);
        dh.alloc(nbytes, kClassBytes);
        std::vector<fp32_t> f = pattern(std::size_t(k), 0.002f, -1.0f, 313);
        std::vector<fp32_t> d = pattern(nelems, 0.004f, 1.5f, 823);
        sc.upload(f, kClassBytes);
        dc.upload(d, kClassBytes);
        sh.upload(f, kClassBytes);
        dh.upload(d, kClassBytes);
        nntile::kernel::add_fiber_inplace::cuda<fp32_t>(s, m, n, k, batch,
            0.75, sh.flat, 2.0, dh.flat);
        nntile::kernel::add_fiber_inplace::chunked<fp32_t>(s, m, n, k, batch,
            0.75, (const fp32_t * const *)sc.table, sc.nchunks, 2.0,
            (fp32_t * const *)dc.table, dc.nchunks, elems(sc), elems(dc),
            cls);
        CK(cudaStreamSynchronize(s));
        check("add_fiber_inplace", dh.gather_flat(), dc.gather(kClassBytes));
    }

    // ---- multiply_fiber: fiber product, all three operands long ----
    {
        Index const m = 1, n = 1, k = nelems;
        Operand s1c, s2c, dc, s1h, s2h, dh;
        s1c.alloc(nbytes, kClassBytes);
        s2c.alloc(nbytes, kClassBytes);
        dc.alloc(nbytes, kClassBytes);
        s1h.alloc(nbytes, kClassBytes);
        s2h.alloc(nbytes, kClassBytes);
        dh.alloc(nbytes, kClassBytes);
        std::vector<fp32_t> f = pattern(nelems, 0.002f, -1.0f, 313);
        std::vector<fp32_t> a = pattern(nelems, 0.004f, 1.5f, 823);
        std::vector<fp32_t> d = pattern(nelems, 0.001f, -0.5f, 127);
        s1c.upload(f, kClassBytes);
        s2c.upload(a, kClassBytes);
        dc.upload(d, kClassBytes);
        s1h.upload(f, kClassBytes);
        s2h.upload(a, kClassBytes);
        dh.upload(d, kClassBytes);
        nntile::kernel::multiply_fiber::cuda<fp32_t>(s, m, n, k, -0.5,
            s1h.flat, s2h.flat, dh.flat);
        nntile::kernel::multiply_fiber::chunked<fp32_t>(s, m, n, k, -0.5,
            (const fp32_t * const *)s1c.table, s1c.nchunks, elems(s1c),
            (const fp32_t * const *)s2c.table, s2c.nchunks, elems(s2c),
            (fp32_t * const *)dc.table, dc.nchunks, elems(dc), cls);
        CK(cudaStreamSynchronize(s));
        check("multiply_fiber", dh.gather_flat(), dc.gather(kClassBytes));
    }

    // ---- sum_fiber: reduction along the middle axis into the fiber ----
    {
        Index const m = 1, n = 1, k = nelems, batch = 1;
        std::size_t const src_bytes = std::size_t(m * n * k * batch)
            * sizeof(fp32_t);
        std::size_t const dst_bytes = std::size_t(k) * sizeof(fp32_t);
        Operand sc, dc, sh, dh;
        sc.alloc(src_bytes, kClassBytes);
        dc.alloc(dst_bytes, kClassBytes);
        sh.alloc(src_bytes, kClassBytes);
        dh.alloc(dst_bytes, kClassBytes);
        std::vector<fp32_t> a = pattern(std::size_t(m * n * k * batch),
            0.002f, -1.0f, 313);
        std::vector<fp32_t> d = pattern(std::size_t(k), 0.0f, 0.5f, 1);
        sc.upload(a, kClassBytes);
        dc.upload(d, kClassBytes);
        sh.upload(a, kClassBytes);
        dh.upload(d, kClassBytes);
        nntile::kernel::sum_fiber::cuda<fp32_t>(s, m, n, k, batch, 0.25,
            sh.flat, 1.5, dh.flat);
        nntile::kernel::sum_fiber::chunked<fp32_t>(s, m, n, k, batch, 0.25,
            (const fp32_t * const *)sc.table, sc.nchunks, 1.5,
            (fp32_t * const *)dc.table, dc.nchunks, elems(sc), elems(dc),
            cls);
        CK(cudaStreamSynchronize(s));
        check("sum_fiber", dh.gather_flat(), dc.gather(kClassBytes));
    }

    // ---- sgd_step: momentum update over long operands ----
    {
        Operand gc, vc, pc, gh, vh, ph;
        gc.alloc(nbytes, kClassBytes);
        vc.alloc(nbytes, kClassBytes);
        pc.alloc(nbytes, kClassBytes);
        gh.alloc(nbytes, kClassBytes);
        vh.alloc(nbytes, kClassBytes);
        ph.alloc(nbytes, kClassBytes);
        std::vector<fp32_t> g = pattern(nelems, 0.002f, -1.0f, 313);
        std::vector<fp32_t> v = pattern(nelems, 0.001f, 0.5f, 127);
        std::vector<fp32_t> pv = pattern(nelems, 0.003f, 2.0f, 719);
        gc.upload(g, kClassBytes);
        vc.upload(v, kClassBytes);
        pc.upload(pv, kClassBytes);
        gh.upload(g, kClassBytes);
        vh.upload(v, kClassBytes);
        ph.upload(pv, kClassBytes);
        nntile::kernel::sgd_step::cuda<fp32_t>(s, 2, nelems, 0.9, 0.01,
            0.001, 0.1, true, gh.flat, vh.flat, ph.flat);
        nntile::kernel::sgd_step::chunked<fp32_t>(s, 2, nelems, 0.9, 0.01,
            0.001, 0.1, true, (const fp32_t * const *)gc.table, gc.nchunks,
            (fp32_t * const *)vc.table, vc.nchunks,
            (fp32_t * const *)pc.table, pc.nchunks,
            elems(gc), elems(vc), elems(pc), cls);
        CK(cudaStreamSynchronize(s));
        check("sgd_step p", ph.gather_flat(), pc.gather(kClassBytes));
    }

    // ---- subtract_indexed_outputs: valid and ignored labels ----
    {
        Index const n_labels = nelems, n_outputs = 2;
        Index const ignore = -100;
        std::size_t const lab_bytes = std::size_t(n_outputs) * sizeof(nntile::int64_t);
        std::size_t const dst_bytes = std::size_t(n_labels * n_outputs)
            * sizeof(fp32_t);
        Operand lc, lh, dc, dh;
        lc.alloc(lab_bytes, kClassBytes);
        lh.alloc(lab_bytes, kClassBytes);
        dc.alloc(dst_bytes, kClassBytes);
        dh.alloc(dst_bytes, kClassBytes);
        std::vector<nntile::int64_t> lab{7, ignore};
        std::vector<fp32_t> d = pattern(std::size_t(n_labels * n_outputs),
            0.004f, 1.5f, 823);
        CK(cudaMemcpy(lc.flat, lab.data(), lab_bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(lc.ptrs[0], lab.data(), lab_bytes,
            cudaMemcpyHostToDevice));
        CK(cudaMemcpy(lh.flat, lab.data(), lab_bytes, cudaMemcpyHostToDevice));
        dc.upload(d, kClassBytes);
        dh.upload(d, kClassBytes);
        nntile::kernel::subtract_indexed_outputs::cuda<fp32_t>(s, n_labels,
            n_outputs, ignore, 0.5,
            (const nntile::int64_t *)lh.flat, dh.flat);
        nntile::kernel::subtract_indexed_outputs::chunked<fp32_t>(s, n_labels,
            n_outputs, ignore, 0.5,
            (const nntile::int64_t * const *)lc.table, lc.nchunks,
            (fp32_t * const *)dc.table, dc.nchunks,
            Index(lab_bytes / sizeof(nntile::int64_t)), elems(dc),
            (const nntile::int64_t *)lh.flat, cls);
        CK(cudaStreamSynchronize(s));
        check("subtract_indexed_outputs", dh.gather_flat(),
            dc.gather(kClassBytes));
    }

    // ---- total_sum_accum: valid and ignored output columns ----
    {
        Index const n_labels = nelems, n_outputs = 2;
        Index const ignore = -100;
        std::size_t const lse_bytes = std::size_t(n_outputs) * sizeof(fp32_t);
        std::size_t const val_bytes = sizeof(fp32_t);
        std::size_t const lab_bytes = std::size_t(n_outputs)
            * sizeof(nntile::int64_t);
        Operand sc, sh, lc, lh, labc, labh, vc1, vh;
        sc.alloc(nbytes, kClassBytes);
        sh.alloc(nbytes, kClassBytes);
        lc.alloc(lse_bytes, kClassBytes);
        lh.alloc(lse_bytes, kClassBytes);
        labc.alloc(lab_bytes, kClassBytes);
        labh.alloc(lab_bytes, kClassBytes);
        vc1.alloc(val_bytes, kClassBytes);
        vh.alloc(val_bytes, kClassBytes);
        std::vector<fp32_t> a = pattern(nelems, 0.002f, -3.0f, 1451);
        fp32_t const zero_v = fp32_t(0.0f);
        sc.upload(a, kClassBytes);
        sh.upload(a, kClassBytes);
        std::vector<fp32_t> lse_h{0.75f, 2.5f}, val_h(1, zero_v);
        std::vector<nntile::int64_t> lab_h{7, ignore};
        lc.upload(lse_h, kClassBytes);
        lh.upload(lse_h, kClassBytes);
        vc1.upload(val_h, kClassBytes);
        vh.upload(val_h, kClassBytes);
        CK(cudaMemcpy(labc.flat, lab_h.data(), lab_bytes,
            cudaMemcpyHostToDevice));
        CK(cudaMemcpy(labc.ptrs[0], lab_h.data(), lab_bytes,
            cudaMemcpyHostToDevice));
        CK(cudaMemcpy(labh.flat, lab_h.data(), lab_bytes,
            cudaMemcpyHostToDevice));
        nntile::kernel::total_sum_accum::cuda<fp32_t>(s, 0.5, n_labels,
            n_outputs, ignore, lh.flat, sh.flat,
            (const nntile::int64_t *)labh.flat, (float *)vh.flat);
        nntile::kernel::total_sum_accum::chunked<fp32_t>(s, 0.5,
            n_labels, n_outputs, ignore,
            (const fp32_t * const *)lc.table, lc.nchunks,
            (const fp32_t * const *)sc.table, sc.nchunks,
            (const nntile::int64_t * const *)labc.table,
            labc.nchunks, (float * const *)vc1.table, vc1.nchunks,
            Index(n_outputs), elems(sc), Index(n_outputs), 1,
            nullptr, nullptr, nullptr, cls);
        CK(cudaStreamSynchronize(s));
        std::vector<fp32_t> const got1 = vc1.gather(kClassBytes);
        std::vector<fp32_t> const ref = vh.gather_flat();
        double const denom = std::max(1.0, std::fabs((double)ref[0].value));
        double const rel = std::fabs((double)got1[0].value
            - (double)ref[0].value) / denom;
        if(rel > 1e-5)
        {
            std::fprintf(stderr, "total_sum_accum: rel diff %g (%f vs %f)\n",
                rel, (double)got1[0].value, (double)ref[0].value);
            ++failures;
        }
        else
        {
            std::printf("total_sum_accum: rel diff %g\n", rel);
        }
    }

    // ---- clear: exact byte range over multiple chunks and a tail ----
    {
        Operand dc;
        dc.alloc(nbytes, kClassBytes);
        std::vector<fp32_t> fill = pattern(nelems, 0.004f, 1.5f, 823);
        dc.upload(fill, kClassBytes);
        nntile::kernel::clear::chunked(s, Index(nbytes),
            (unsigned char * const *)dc.table, dc.nchunks,
            Index(kClassBytes), cls);
        CK(cudaStreamSynchronize(s));
        std::vector<fp32_t> const zero(nelems, fp32_t(0.0f));
        check("clear", zero, dc.gather(kClassBytes));
    }

    // ---- mask_scalar: bool mask, both operands multi-chunk ----
    {
        Index const nrows = nelems, ncols = 1;
        std::size_t const mask_bytes = std::size_t(nrows) * sizeof(bool_t);
        Operand dc, dh;
        dc.alloc(nbytes, kClassBytes);
        dh.alloc(nbytes, kClassBytes);
        std::vector<fp32_t> a = pattern(nelems, 0.006f, -13.0f, 1103);
        std::vector<bool_t> mask;
        mask.resize(std::size_t(nrows));
        for(Index i = 0; i < nrows; ++i)
        {
            mask[std::size_t(i)] = bool_t(i % 3 != 0);
        }
        dc.upload(a, kClassBytes);
        dh.upload(a, kClassBytes);
        // The mask has its own chunk chain, laid out with the same class.
        Operand mask_c, mask_h;
        mask_c.alloc(mask_bytes, kClassBytes);
        mask_h.alloc(mask_bytes, kClassBytes);
        CK(cudaMemcpy(mask_h.flat, mask.data(), mask_bytes,
            cudaMemcpyHostToDevice));
        std::size_t const mask_class = kClassBytes;
        for(Index c = 0; c < mask_c.nchunks; ++c)
        {
            std::size_t const off = std::size_t(c) * mask_class;
            Index const len = std::min<Index>(Index(mask_class),
                Index(mask_bytes - off));
            CK(cudaMemcpy(mask_c.ptrs[c], mask.data() + off,
                std::size_t(len), cudaMemcpyHostToDevice));
        }
        nntile::kernel::mask_scalar::cuda<fp32_t>(s, nrows, ncols,
            (bool_t const *)mask_h.flat, -9.5, dh.flat);
        nntile::kernel::mask_scalar::chunked<fp32_t>(s, nrows, ncols, -9.5,
            (fp32_t * const *)dc.table, dc.nchunks, chunk_elems,
            (bool_t const * const *)mask_c.table, mask_c.nchunks,
            Index(mask_class / sizeof(bool_t)), cls);
        CK(cudaStreamSynchronize(s));
        check("mask_scalar", dh.gather_flat(), dc.gather(kClassBytes));
    }

    CK(cudaStreamDestroy(s));
    if(failures != 0)
    {
        std::fprintf(stderr, "chunked_ops: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("chunked_ops: PASS\n");
    return 0;
}
