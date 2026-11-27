/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file nntile/tests/kernel/chunked_ab.cc
 * Kernel-level A/B: the classic contiguous entry point against the chunked
 * entry point on the same bytes, including a non-class-size tail. Built and
 * run in BOTH backend configurations: the chunked entry points are
 * backend-agnostic (with a 16 MiB default class when nnhaul is absent).
 *
 * @version 1.1.0
 * */

#include <nntile/kernel/add/cuda.hh>
#include <nntile/kernel/copy/cuda.hh>
#include <nntile/kernel/transpose/cuda.hh>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using nntile::Index;
using nntile::fp32_t;
using nntile::kernel::ChunkClass;

namespace
{

#define CK(x) do { cudaError_t e_ = (x); if(e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
        cudaGetErrorString(e_)); std::exit(2); } } while(0)

//! A chunked operand: n chunks of `cls.bytes`, last one shorter.
struct Chunks
{
    std::vector<void *> ptrs;
    void **table = nullptr;
    Index n = 0;

    void alloc(std::size_t nbytes, ChunkClass cls)
    {
        std::size_t const chunk = cls.mask + 1;
        n = Index((nbytes + chunk - 1) / chunk);
        ptrs.resize(n);
        for(Index i = 0; i < n; ++i)
        {
            std::size_t const off = i * chunk;
            std::size_t const len = std::min(chunk, nbytes - off);
            CK(cudaMalloc(&ptrs[i], len));
        }
        CK(cudaMalloc(&table, n * sizeof(void *)));
        CK(cudaMemcpy(table, ptrs.data(), n * sizeof(void *),
            cudaMemcpyHostToDevice));
    }
};

int failures = 0;

void check(char const *name, std::vector<fp32_t> const &a,
    std::vector<fp32_t> const &b)
{
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

} // namespace

int main()
{
    // 10 chunks of 64 KiB plus a 3-word tail.
    constexpr std::size_t kClassBytes = 64u << 10;
    ChunkClass const cls = make_class(kClassBytes);
    std::size_t const elems_per_chunk = kClassBytes / sizeof(fp32_t);
    Index const nelems = Index(10 * elems_per_chunk + 3);
    std::size_t const nbytes = nelems * sizeof(fp32_t);

    std::vector<fp32_t> h1(nelems), h2(nelems), href(nelems), hgot(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        h1[i] = fp32_t(0.5f * float(i % 97) + 1.0f);
        h2[i] = fp32_t(-0.25f * float(i % 53) + 2.0f);
    }

    cudaStream_t s;
    CK(cudaStreamCreate(&s));

    // ---- add: cuda() versus chunked() ----
    {
        Chunks c1, c2, cd;
        c1.alloc(nbytes, cls);
        c2.alloc(nbytes, cls);
        cd.alloc(nbytes, cls);
        for(Index c = 0; c < c1.n; ++c)
        {
            std::size_t const off = std::size_t(c) * elems_per_chunk;
            Index const len = std::min<Index>(Index(elems_per_chunk),
                nelems - Index(off));
            CK(cudaMemcpy(c1.ptrs[c], h1.data() + off, len * sizeof(fp32_t),
                cudaMemcpyHostToDevice));
            CK(cudaMemcpy(c2.ptrs[c], h2.data() + off, len * sizeof(fp32_t),
                cudaMemcpyHostToDevice));
        }
        fp32_t *a, *b, *r;
        CK(cudaMalloc(&a, nbytes));
        CK(cudaMalloc(&b, nbytes));
        CK(cudaMalloc(&r, nbytes));
        CK(cudaMemcpy(a, h1.data(), nbytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b, h2.data(), nbytes, cudaMemcpyHostToDevice));

        nntile::kernel::add::cuda<fp32_t>(s, nelems, 1.0, a, 1.0, b, r);
        nntile::kernel::add::chunked<fp32_t>(s, nelems, 1.0,
            (const fp32_t * const *)c1.table, c1.n, 1.0,
            (const fp32_t * const *)c2.table, c2.n,
            (fp32_t * const *)cd.table, cd.n,
            Index(elems_per_chunk), Index(elems_per_chunk),
            Index(elems_per_chunk), cls);
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(href.data(), r, nbytes, cudaMemcpyDeviceToHost));
        for(Index c = 0; c < cd.n; ++c)
        {
            std::size_t const off = std::size_t(c) * elems_per_chunk;
            Index const len = std::min<Index>(Index(elems_per_chunk),
                nelems - Index(off));
            CK(cudaMemcpy(hgot.data() + off, cd.ptrs[c],
                len * sizeof(fp32_t), cudaMemcpyDeviceToHost));
        }
        check("add", href, hgot);
        CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(r));
    }

    // ---- copy: cuda() versus chunked() ----
    {
        Chunks src, dst;
        src.alloc(nbytes, cls);
        dst.alloc(nbytes, cls);
        for(Index c = 0; c < src.n; ++c)
        {
            std::size_t const off = std::size_t(c) * elems_per_chunk;
            Index const len = std::min<Index>(Index(elems_per_chunk),
                nelems - Index(off));
            CK(cudaMemcpy(src.ptrs[c], h1.data() + off, len * sizeof(fp32_t),
                cudaMemcpyHostToDevice));
        }
        nntile::kernel::copy::chunked(s, nbytes,
            (const unsigned char * const *)src.table, src.n,
            (unsigned char * const *)dst.table, dst.n,
            Index(kClassBytes), Index(kClassBytes), cls);
        CK(cudaStreamSynchronize(s));
        for(Index c = 0; c < dst.n; ++c)
        {
            std::size_t const off = std::size_t(c) * elems_per_chunk;
            Index const len = std::min<Index>(Index(elems_per_chunk),
                nelems - Index(off));
            CK(cudaMemcpy(hgot.data() + off, dst.ptrs[c],
                len * sizeof(fp32_t), cudaMemcpyDeviceToHost));
        }
        check("copy", h1, hgot);
    }

    // ---- transpose: cuda() versus chunked() ----
    {
        constexpr Index m = 1000, n = 1000;
        Index const total = m * n;
        std::vector<fp32_t> src_h(total), ref_h(total), got_h(total);
        for(Index i = 0; i < total; ++i)
        {
            src_h[i] = fp32_t(0.25f * float(i % 211) - 3.0f);
        }
        std::size_t const src_bytes = std::size_t(total) * sizeof(fp32_t);
        Chunks cs, cdst;
        cs.alloc(src_bytes, cls);
        cdst.alloc(src_bytes, cls);
        for(Index c = 0; c < cs.n; ++c)
        {
            std::size_t const off = std::size_t(c) * elems_per_chunk;
            Index const len = std::min<Index>(Index(elems_per_chunk),
                total - Index(off));
            CK(cudaMemcpy(cs.ptrs[c], src_h.data() + off,
                len * sizeof(fp32_t), cudaMemcpyHostToDevice));
        }
        fp32_t *src_d, *ref_d;
        CK(cudaMalloc(&src_d, src_bytes));
        CK(cudaMalloc(&ref_d, src_bytes));
        CK(cudaMemcpy(src_d, src_h.data(), src_bytes,
            cudaMemcpyHostToDevice));
        nntile::kernel::transpose::cuda<fp32_t>(s, m, n, 1.0, src_d, ref_d);
        nntile::kernel::transpose::chunked<fp32_t>(s, m, n, 1.0,
            (const fp32_t * const *)cs.table, cs.n,
            (fp32_t * const *)cdst.table, cdst.n,
            Index(elems_per_chunk), Index(elems_per_chunk), cls);
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(ref_h.data(), ref_d, src_bytes,
            cudaMemcpyDeviceToHost));
        for(Index c = 0; c < cdst.n; ++c)
        {
            std::size_t const off = std::size_t(c) * elems_per_chunk;
            Index const len = std::min<Index>(Index(elems_per_chunk),
                total - Index(off));
            CK(cudaMemcpy(got_h.data() + off, cdst.ptrs[c],
                len * sizeof(fp32_t), cudaMemcpyDeviceToHost));
        }
        check("transpose", ref_h, got_h);
        CK(cudaFree(src_d)); CK(cudaFree(ref_d));
    }

    CK(cudaStreamDestroy(s));
    if(failures != 0)
    {
        std::fprintf(stderr, "chunked_ab: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("chunked_ab: PASS\n");
    return 0;
}
