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

constexpr std::size_t kChunkElems = std::size_t(1) << 20; // 4 MiB fp32

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
    std::uint32_t &rng)
{
    int const nchunks =
        int((host.size() + kChunkElems - 1) / kChunkElems);
    std::vector<float *> htab(nchunks);
    for(int c = 0; c < nchunks; ++c)
    {
        if((lcg(rng) & 1) != 0)
        {
            void *j = nullptr;
            if(cudaMalloc(&j, kChunkElems / 8) == cudaSuccess)
            {
                da.ptrs.push_back(j);
            }
        }
        std::size_t const off = std::size_t(c) * kChunkElems;
        std::size_t const len =
            std::min(kChunkElems, host.size() - off);
        float *d = nullptr;
        TEST_ASSERT(cudaMalloc(&d, kChunkElems * sizeof(float))
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

    DevAllocs da;
    std::uint32_t rng = 99;
    Chunked const ca = make_chunked(ha, da, rng);
    Chunked const cb = make_chunked(hb, da, rng);
    Chunked const cc = make_chunked(hc, da, rng);

    Operand const oa_s{ca.table, 20};
    Operand const ob_s{cb.table, 20};
    Operand const oc_s{(void *)cc.table, 20};
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
    for(int c = 0, nch = int((hc.size() + kChunkElems - 1) / kChunkElems);
        c < nch; ++c)
    {
        std::size_t const off = std::size_t(c) * kChunkElems;
        std::size_t const len = std::min(kChunkElems, hc.size() - off);
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
    std::printf("gemm_chunked %-34s m=%5d n=%5d k=%5d b=%d max diff %.2e"
        " (tol %.2e)\n",
        label, m, n, k, batch, worst, tol);
}

} // namespace

int main()
{
    // Ownership policy: the measured families own their calls, the
    // rest decline (the staged path stays in charge there).
    TEST_ASSERT(!kernel::gemm_chunked::launch(
        Op::NoTrans, Op::NoTrans, 4096, 768, 3072, 1,
        1.0f, Operand{nullptr, 0}, Operand{nullptr, 0}, 0.0f,
        Operand{nullptr, 0}, 0, 0, 0, 0));
    // Tails below the 128x128x32 tile, every transpose combo.
    run_case(Op::NoTrans, Op::NoTrans, 8, 12, 20, 1, 0, 0, "NN tiny");
    run_case(Op::NoTrans, Op::Trans, 16, 8, 24, 1, 0, 0, "NT tiny");
    run_case(Op::Trans, Op::NoTrans, 12, 8, 16, 1, 0, 0, "TN tiny");
    run_case(Op::Trans, Op::Trans, 8, 8, 8, 1, 0, 0, "TT tiny");
    // Multi-tile with M/N/K tails.
    run_case(Op::NoTrans, Op::NoTrans, 300, 260, 516, 1, 0, 0, "NN mid");
    run_case(Op::NoTrans, Op::Trans, 260, 300, 772, 1, 0, 0, "NT mid");
    run_case(Op::Trans, Op::NoTrans, 300, 132, 516, 1, 0, 0, "TN mid");
    // Straddle-heavy: rows of 1000 floats cross the 4 MiB class at
    // 1048.576 rows (the embedding geometry).
    run_case(Op::NoTrans, Op::NoTrans, 1200, 64, 1000, 1, 0, 0,
        "NN straddle");
    run_case(Op::NoTrans, Op::Trans, 1200, 64, 1000, 1, 0, 0,
        "NT straddle");
    // Odd N (lm_head's vocab): the ownership law declines it — the
    // staged cuBLAS path owns misaligned shapes.
    TEST_ASSERT(!kernel::gemm_chunked::supports(Op::NoTrans, Op::Trans,
        260, 301, 128, 1));
    TEST_ASSERT(!kernel::gemm_chunked::supports(Op::Trans, Op::NoTrans,
        50257, 768, 4096, 1));
    // Batching and broadcast.
    run_case(Op::NoTrans, Op::NoTrans, 40, 64, 96, 3, 0, 0, "NN batch");
    run_case(Op::NoTrans, Op::NoTrans, 40, 64, 96, 3, 0, 1, "NN bcast-b");
    run_case(Op::NoTrans, Op::NoTrans, 40, 64, 96, 3, 1, 0, "NN bcast-a");
    return 0;
}

#else

int main()
{
    return 0;
}

#endif
