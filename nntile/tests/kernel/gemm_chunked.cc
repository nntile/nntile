/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/gemm_chunked.cc
 * Native chunked fp32 GEMM kernel vs a CPU reference, over the three
 * operand addressing modes (flat, class table, one-shot sentinel),
 * straddle-heavy geometries, tails, batching and broadcast.
 *
 * @version 1.0.0
 * */

#include "nntile/base_types.hh"
#include "nntile/kernel/gemm_chunked/cuda.hh"
#include "../testing.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace nntile;

#ifdef NNTILE_USE_CUDA

using kernel::gemm_chunked::Operand;
using kernel::gemm_chunked::Op;

namespace
{

//! One chunk-class configuration: power-of-two element count and its
//! log2 (the Operand shift). The suite sweeps the supported classes:
//! any power of two from 1 MiB upward (the engine's compile-time class
//! law, NNHAUL_CHUNK_SIZE_MB).
struct ChunkClass
{
    std::size_t elems;
    int shift;
};

ChunkClass const kClasses[] = {
    {std::size_t(1) << 18, 18}, // 1 MiB
    {std::size_t(1) << 19, 19}, // 2 MiB
    {std::size_t(1) << 20, 20}, // 4 MiB (engine default)
    {std::size_t(1) << 21, 21}, // 8 MiB
};

std::uint32_t lcg(std::uint32_t &s)
{
    s = s * 1664525u + 1013904223u;
    return s >> 8;
}

void fill(std::vector<float> &v, std::uint32_t seed)
{
    std::uint32_t s = seed;
    for(auto &x : v)
    {
        x = (int(lcg(s) % 2001) - 1000) / 2000.0f; // [-0.5, 0.5]
    }
}

//! C(m,n) = alpha*op(A)*op(B) + beta*C, C-order row-major, storage per
//! the kernel contract: A [m,k] (NoTrans) / [k,m] (Trans); B [k,n] /
//! [n,k]; C [m,n]; per-batch element strides.
void ref_gemm(Op ta, Op tb, int m, int n, int k, int batch,
    float alpha, std::vector<float> const &a, std::vector<float> const &b,
    float beta, std::vector<float> &c, bool bcast_a, bool bcast_b)
{
    long long const sa = bcast_a ? 0 : (long long)m * k;
    long long const sb = bcast_b ? 0 : (long long)k * n;
    long long const sc = (long long)m * n;
    for(int bi = 0; bi < batch; ++bi)
    {
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                float acc = 0.0f;
                for(int l = 0; l < k; ++l)
                {
                    float const av = ta == Op::NoTrans
                        ? a[bi * sa + (long long)i * k + l]
                        : a[bi * sa + (long long)l * m + i];
                    float const bv = tb == Op::NoTrans
                        ? b[bi * sb + (long long)l * n + j]
                        : b[bi * sb + (long long)j * k + l];
                    acc += av * bv;
                }
                float &out = c[bi * sc + (long long)i * n + j];
                out = alpha * acc + beta * out;
            }
        }
    }
}

struct DevAllocs
{
    std::vector<void *> ptrs;
    ~DevAllocs()
    {
        for(void *p : ptrs)
        {
            cudaFree(p);
        }
    }
};

//! Chunk-scattered device copy of `host` plus its device table (the
//! ring's regime); junk allocations interleave the chunks.
struct Chunked
{
    float *const *table = nullptr;
};

Chunked make_chunked(std::vector<float> const &host, DevAllocs &da,
    std::uint32_t &rng, ChunkClass cc)
{
    int const nchunks =
        int((host.size() + cc.elems - 1) / cc.elems);
    std::vector<float *> htab(nchunks);
    for(int c = 0; c < nchunks; ++c)
    {
        if((lcg(rng) & 1) != 0)
        {
            void *j = nullptr;
            if(cudaMalloc(&j, cc.elems / 8) == cudaSuccess)
            {
                da.ptrs.push_back(j);
            }
        }
        std::size_t const off = std::size_t(c) * cc.elems;
        std::size_t const len =
            std::min(cc.elems, host.size() - off);
        float *d = nullptr;
        TEST_ASSERT(cudaMalloc(&d, cc.elems * sizeof(float))
            == cudaSuccess);
        TEST_ASSERT(cudaMemcpy(d, host.data() + off, len * sizeof(float),
            cudaMemcpyHostToDevice) == cudaSuccess);
        da.ptrs.push_back(d);
        htab[c] = d;
    }
    float *const *tab = nullptr;
    TEST_ASSERT(cudaMalloc((void **)&tab,
                  nchunks * sizeof(float *)) == cudaSuccess);
    TEST_ASSERT(cudaMemcpy((void *)tab, (void const *)htab.data(),
                  nchunks * sizeof(float *),
                  cudaMemcpyHostToDevice) == cudaSuccess);
    da.ptrs.push_back((void *)tab);
    Chunked r;
    r.table = tab;
    return r;
}

void run_case(Op ta, Op tb, int m, int n, int k, int batch, bool bcast_a,
    bool bcast_b, const char *label)
{
    long long const a_store = ta == Op::NoTrans ? (long long)m * k
                                                : (long long)k * m;
    long long const b_store = tb == Op::NoTrans ? (long long)k * n
                                                : (long long)n * k;
    long long const c_store = (long long)m * n;
    int const a_copies = bcast_a ? 1 : batch;
    int const b_copies = bcast_b ? 1 : batch;

    std::vector<float> ha(size_t(a_store) * a_copies);
    std::vector<float> hb(size_t(b_store) * b_copies);
    std::vector<float> hc(size_t(c_store) * batch);
    fill(ha, 1 + size_t(m) * 131);
    fill(hb, 7 + size_t(n) * 197);
    fill(hc, 3 + size_t(k) * 89);

    std::vector<float> const c_ref(hc);
    ref_gemm(ta, tb, m, n, k, batch, 0.7f, ha, hb, 0.3f,
        const_cast<std::vector<float> &>(c_ref), bcast_a, bcast_b);

    if(!kernel::gemm_chunked::supports(ta, tb, m, n, k, batch))
    {
        std::printf("gemm_chunked %s: unsupported (fallback owns it)\n",
            label);
        return;
    }

    for(ChunkClass cls : kClasses)
    {
    DevAllocs da;
    std::uint32_t rng = 99;
    Chunked const ca = make_chunked(ha, da, rng, cls);
    Chunked const cb = make_chunked(hb, da, rng, cls);
    Chunked const cc = make_chunked(hc, da, rng, cls);

    Operand const oa_s{ca.table, cls.shift};
    Operand const ob_s{cb.table, cls.shift};
    Operand const oc_s{(void *)cc.table, cls.shift};
    long long const sa = bcast_a ? 0LL : a_store;
    long long const sb = bcast_b ? 0LL : b_store;
    // Kernel correctness: force the best config regardless of the
    // ownership policy (launch() alone would decline shapes the staged
    // path still owns).
    bool const ok = kernel::gemm_chunked::launch_forced(ta, tb, m, n, k,
        batch, 0.7f, oa_s, ob_s, 0.3f, oc_s, sa, sb, c_store, 0);
    TEST_ASSERT(ok);
    TEST_ASSERT(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> got(hc.size());
    for(int c = 0, nch = int((hc.size() + cls.elems - 1) / cls.elems);
        c < nch; ++c)
    {
        std::size_t const off = std::size_t(c) * cls.elems;
        std::size_t const len = std::min(cls.elems, hc.size() - off);
        float *src = nullptr;
        TEST_ASSERT(cudaMemcpy(&src,
                       (char const *)cc.table + c * sizeof(float *),
                       sizeof(float *), cudaMemcpyDeviceToHost)
            == cudaSuccess);
        TEST_ASSERT(cudaMemcpy(got.data() + off, src, len * sizeof(float),
            cudaMemcpyDeviceToHost) == cudaSuccess);
    }

    double worst = 0.0;
    for(std::size_t i = 0; i < got.size(); ++i)
    {
        // NaN-blind-proof: an explicit NaN check, then the gap (a bare
        // max()/comparison silently skips NaN diffs).
        if(std::isnan(got[i]) || std::isnan(c_ref[i]))
        {
            std::fprintf(stderr, "gemm_chunked %s: NaN at %zu\n", label,
                i);
            throw std::runtime_error("gemm_chunked NaN");
        }
        double const d = std::fabs(double(got[i]) - double(c_ref[i]));
        if(!(d <= worst) || d > worst)
        {
            worst = d > worst ? d : worst;
        }
    }
    double const tol = 16.0 * std::numeric_limits<float>::epsilon() * k
        * 0.0625 + 1e-6;
    if(worst > tol)
    {
        std::fprintf(stderr,
            "gemm_chunked %s: max diff %g exceeds tol %g\n", label, worst,
            tol);
        throw std::runtime_error("gemm_chunked mismatch");
    }
    std::printf("gemm_chunked %-28s 2^%-2d m=%5d n=%5d k=%5d b=%d"
        " max diff %.2e (tol %.2e)\n",
        label, cls.shift, m, n, k, batch, worst, tol);
    }
}

} // namespace

int main()
{
    // FINAL shape law (owner directive): each dim exactly 64 or a
    // 128-multiple; everything else declines to staged cuBLAS.
    TEST_ASSERT(!kernel::gemm_chunked::supports(Op::NoTrans, Op::Trans,
        260, 301, 128, 1)); // odd n
    TEST_ASSERT(!kernel::gemm_chunked::supports(Op::Trans, Op::NoTrans,
        50257, 768, 4096, 1)); // odd m (lm_head wgrad)
    TEST_ASSERT(!kernel::gemm_chunked::supports(Op::NoTrans, Op::NoTrans,
        96, 128, 256, 1)); // 96 is 1.5x64 -- no clause
    TEST_ASSERT(!kernel::gemm_chunked::supports(Op::NoTrans, Op::NoTrans,
        4096, 192, 3072, 1)); // 192 is 1.5x128 -- no clause
    TEST_ASSERT(kernel::gemm_chunked::supports(Op::NoTrans, Op::NoTrans,
        4096, 384, 3072, 1)); // 384 = 3x128 (was declined under the
    // round-8 interim law)
    TEST_ASSERT(kernel::gemm_chunked::supports(Op::NoTrans, Op::NoTrans,
        4096, 1536, 3072, 1)); // 12x128
    TEST_ASSERT(!kernel::gemm_chunked::supports(Op::NoTrans, Op::NoTrans,
        4096, 1600, 3072, 1)); // GPT-2 XL hidden = 12.5x128: staged
    // path owns it under the final law
    TEST_ASSERT(kernel::gemm_chunked::supports(Op::NoTrans, Op::NoTrans,
        4096, 64, 64, 12)); // attention: exact 64s
    TEST_ASSERT(kernel::gemm_chunked::supports(Op::NoTrans, Op::NoTrans,
        128, 64, 1024, 1)); // single-batch 64 dims (MsSmall routing)

    // Classic family (launch_forced): 128-multiple shapes, every
    // transpose combo, deep-k, heavy multi-chunk fragmentation, and
    // chunk-straddle geometry -- all scattered over 1/2/4/8 MiB
    // classes with junk-interleaved non-contiguous chunk tables.
    run_case(Op::NoTrans, Op::NoTrans, 256, 512, 1024, 1, 0, 0,
        "NN law");
    run_case(Op::NoTrans, Op::Trans, 512, 384, 768, 1, 0, 0,
        "NT 128n");
    run_case(Op::Trans, Op::NoTrans, 384, 256, 1024, 1, 0, 0,
        "TN 128n");
    run_case(Op::Trans, Op::Trans, 128, 128, 256, 1, 0, 0, "TT law");
    run_case(Op::NoTrans, Op::NoTrans, 512, 768, 3072, 1, 0, 0,
        "NN deep-k");
    // Fragmentation-heavy: every operand spans several 1 MiB chunks at
    // the smallest class; rows march across chunk boundaries.
    run_case(Op::NoTrans, Op::NoTrans, 1152, 384, 1280, 1, 0, 0,
        "NN frag");
    run_case(Op::NoTrans, Op::Trans, 1280, 1152, 384, 1, 0, 0,
        "NT frag");
    // Straddle-heavy: 1024-float rows cross the 4 MiB class at row
    // 1024; m=1280 covers the boundary.
    run_case(Op::NoTrans, Op::NoTrans, 1280, 128, 1024, 1, 0, 0,
        "NN straddle");
    run_case(Op::NoTrans, Op::Trans, 1280, 128, 1024, 1, 0, 0,
        "NT straddle");
    // Small-tile family: batched and single-batch 64-dim calls.
    run_case(Op::NoTrans, Op::NoTrans, 128, 64, 128, 3, 0, 0,
        "NN batch");
    run_case(Op::NoTrans, Op::NoTrans, 128, 64, 128, 3, 0, 1,
        "NN bcast-b");
    run_case(Op::NoTrans, Op::NoTrans, 128, 64, 128, 3, 1, 0,
        "NN bcast-a");
    run_case(Op::NoTrans, Op::NoTrans, 128, 64, 1024, 1, 0, 0,
        "NN sb64");
    return 0;
}

#else

int main()
{
    return 0;
}

#endif
