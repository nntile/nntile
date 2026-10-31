/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile/chunked_big.cc
 * Big-tile graph-path chunking test: structured ops on 2 MiB tiles
 * allocated by Runtime::allocate_missing_tiles, exercised in two chunk
 * ladders (the default and a deep 4096-byte one via a child process),
 * with a kernel-time chunk_count probe proving the chunked layout was
 * actually live.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "context_fixture.hh"

#include <nntile/constants.hh>
#include <nntile/core/tile.hh>
#include <nntile/kernel/maxsumexp.hh>
#include <nntile/kernel/norm_slice_inplace.hh>
#include <nntile/kernel/softmax_inplace.hh>
#include <nntile/kernel/transpose.hh>
#include <nntile/nnhaul/ops/gemm.hh>
#include <nntile/nnhaul/stage.hh>
#include <nntile/tile.hh>
#include <nntile/tile/ops/embedding.hh>
#include <nntile/tile/ops/embedding_backward.hh>
#include <nntile/tile/ops/gemm.hh>
#include <nntile/tile/ops/maxsumexp.hh>
#include <nntile/tile/ops/norm_slice_inplace.hh>
#include <nntile/tile/ops/softmax_inplace.hh>
#include <nntile/tile/ops/transpose.hh>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>

#ifdef NNTILE_USE_CBLAS
#include <nntile/kernel/cblas.hh>
#endif

using namespace nntile;
namespace tg = nntile::tile;

namespace
{

// Chunk-count probe: a trivial codelet whose kernel records the chunk
// table size the runtime handed it. This is the ground-truth proof that
// the tile's device replica is actually chunked (not just that the
// allocation policy said so).
std::atomic<std::size_t> g_probe_chunks{0};

void probe_chunks_cuda(void *buffers[], void *) noexcept
{
    g_probe_chunks.store(
        static_cast<::nnhaul::Buffer *>(buffers[0])->chunk_count());
}

std::size_t probe_tile_chunks(std::size_t nbytes)
{
    ::nnhaul::Codelet probe("chunk_probe", nullptr, probe_chunks_cuda,
        nullptr);
    nntile::core::Tile<fp32_t> tile({static_cast<Index>(nbytes / 4)});
    {
        auto held = tile.handle().acquire(::nnhaul::Access::Write);
        for (Index i = 0; i < tile.nelems; ++i)
        {
            held.get_ptr<fp32_t>()[i] = fp32_t(0.0f);
        }
    }
    ::nnhaul::insert(probe, {{&tile.handle(), ::nnhaul::Access::Read}},
        nullptr, 0);
    ::nnhaul::wait();
    return g_probe_chunks.load();
}

float max_rel_diff(std::vector<float> const &got, std::vector<fp32_t> const &ref)
{
    REQUIRE(got.size() == ref.size());
    float worst = 0.0f;
    for (std::size_t i = 0; i < got.size(); ++i)
    {
        float const r = static_cast<float>(ref[i]);
        float const g = got[i];
        float const scale = std::max(std::max(std::abs(r), std::abs(g)), 1.0f);
        float const rel = std::abs(g - r) / scale;
        worst = std::max(worst, rel);
    }
    return worst;
}

// The big-tile graph: gemm -> maxsumexp -> softmax_inplace (in-place on
// the gemm output) -> transpose -> norm_slice_inplace. A, C and T are
// 2 MiB fp32 tiles, B is 1 MiB, so every operand sits above the
// chunking threshold of every ladder this test runs under.
//
// m = 1024, n = 512, k = 512 (fp32): A = 2 MiB, B = 1 MiB, C = 2 MiB,
// T = 2 MiB. The reference is a plain host computation (cblas gemm plus
// the cpu kernels) on the same inputs.
void run_big_tile_chunked_graph()
{
    Index const m = 1024, n = 512, k = 512;
    std::size_t const nbytes_big = sizeof(float) * m * k;
    REQUIRE(nbytes_big == std::size_t{2} << 20);

    bool const chunking_off = false; // one class: every tile is chunked
    if (chunking_off)
    {
        std::cout << "chunked_big: NNHAUL_CHUNK_TILES=0 - engagement "
                     "assertions skipped, graph still runs\n";
    }
    else
    {
        // Engagement proof 1: a same-size tile's device buffer really is
        // chunked, with exactly the ladder's chunk count.
        std::size_t const expect_chunks =
            (nbytes_big + ::nnhaul::kChunkClassBytes - 1)
            / ::nnhaul::kChunkClassBytes;
        std::size_t const got_chunks = probe_tile_chunks(nbytes_big);
        INFO("probe chunk_count=" << got_chunks << " expected "
                                  << expect_chunks);
        REQUIRE(got_chunks == expect_chunks);
        REQUIRE(got_chunks > 1);
    }

    std::vector<float> a(static_cast<std::size_t>(m * k));
    std::vector<float> b(static_cast<std::size_t>(n * k));
    for (Index i = 0; i < m * k; ++i)
    {
        a[static_cast<std::size_t>(i)] =
            0.01f * static_cast<float>((i % 17) - 8);
    }
    for (Index i = 0; i < n * k; ++i)
    {
        b[static_cast<std::size_t>(i)] =
            0.01f * static_cast<float>((i % 13) - 6);
    }
    std::vector<fp32_t> af(a.begin(), a.end());
    std::vector<fp32_t> bf(b.begin(), b.end());

    // Host reference pipeline. Kernel (m, n, k) convention: m-by-k-by-n
    // with the MIDDLE axis k reduced; a tile-axis-0 reduction of an
    // {r, c} tile maps to (m=c, n=1, k=r).
    std::vector<fp32_t> c_ref(static_cast<std::size_t>(m * n), fp32_t(0.0f));
    // The haul tile layer stores tiles C-order (row-major, last axis
    // contiguous; the kernel [m, k, n] map's m is the CONTIGUOUS
    // extent). This reference walks the same convention: A {m, k} at
    // i*k + l, B {k, n} at l*n + j, C {m, n} at i*n + j. The previous
    // col-major strides computed a different product of the same
    // buffers (the tile layer predates this test's fix); only the
    // maxsumexp comparison was sharp enough to expose it — the
    // gemm+softmax check's scale is floored at 1.0 and softmax over
    // 1024 near-uniform elements cannot move it.
    for (Index i = 0; i < m; ++i)
    {
        for (Index j = 0; j < n; ++j)
        {
            float acc = 0.0f;
            for (Index l = 0; l < k; ++l)
            {
                acc += a[static_cast<std::size_t>(i) * k + l]
                    * b[static_cast<std::size_t>(l) * n + j];
            }
            c_ref[static_cast<std::size_t>(i) * n + j] = fp32_t(acc);
        }
    }
    std::vector<fp32_t> mse_ref(static_cast<std::size_t>(2 * n), fp32_t(0.0f));
    nntile::kernel::maxsumexp::cpu<fp32_t>(
        n, Index{1}, m, c_ref.data(), Scalar{0.0}, mse_ref.data());
    nntile::kernel::softmax_inplace::cpu<fp32_t>(
        n, Index{1}, m, mse_ref.data(), Scalar{1.0}, c_ref.data());
    std::vector<fp32_t> t_ref(static_cast<std::size_t>(n * m), fp32_t(0.0f));
    // The tile op passes (src.shape[1], src.shape[0]) as the kernel's
    // (m, n) - see core/transpose.cc.
    nntile::kernel::transpose::cpu<fp32_t>(
        n, m, Scalar{1.0}, c_ref.data(), t_ref.data());
    std::vector<fp32_t> d_ref(static_cast<std::size_t>(m), fp32_t(0.0f));
    nntile::kernel::norm_slice_inplace::cpu<fp32_t>(
        m, Index{1}, n, Scalar{1.0}, t_ref.data(), Scalar{0.0},
        d_ref.data());

    // The graph.
    TileGraph graph("chunked_big");
    auto *ta = graph.data({m, k}, "a", DataType::FP32);
    auto *tb = graph.data({k, n}, "b", DataType::FP32);
    auto *tc = graph.data({m, n}, "c", DataType::FP32);
    auto *tmse = graph.data({n, 2}, "mse", DataType::FP32);
    auto *tt = graph.data({n, m}, "t", DataType::FP32);
    auto *td = graph.data({m}, "d", DataType::FP32);

    tg::gemm(ta, tb, tc, Scalar{1.0}, Scalar{0.0}, false, false, Index{1},
        Index{0});
    tg::maxsumexp(tc, tmse, Index{0}, Scalar{0.0}, 0);
    tg::softmax_inplace(tmse, Scalar{1.0}, tc, Index{0});
    tg::transpose(Scalar{1.0}, tc, tt, Index{1});
    tg::norm_slice_inplace(Scalar{1.0}, tt, Scalar{0.0}, td, Index{0}, 0);

    std::uint64_t const staged_before =
        chunking_off ? 0 : ::nntile::haul::stage::staged_submit_count();
    std::uint64_t const native_before =
        chunking_off ? 0 : ::nntile::haul::gemm_native_chunked_count();
    Runtime runtime(graph);
    runtime.compile();
    runtime.bind_data(ta, a);
    runtime.bind_data(tb, b);
    std::vector<float> c_init(static_cast<std::size_t>(m * n), 0.0f);
    runtime.bind_data(tc, c_init);
    std::vector<float> mse_init(static_cast<std::size_t>(2 * n), 0.0f);
    runtime.bind_data(tmse, mse_init);
    std::vector<float> t_init(static_cast<std::size_t>(n * m), 0.0f);
    runtime.bind_data(tt, t_init);
    std::vector<float> d_init(static_cast<std::size_t>(m), 0.0f);
    runtime.bind_data(td, d_init);
    runtime.execute();
    runtime.wait();

    if (!chunking_off)
    {
        // Engagement proof 2: gemm runs the NATIVE chunked panel driver,
        // and after the family-1 promotion so do maxsumexp,
        // softmax_inplace and norm_slice_inplace (the chunked unit
        // drivers), and after the family-2 promotion so does the
        // transpose (the chunked panel driver) - NO staged submit is
        // left in the chain.
        std::uint64_t const staged_after =
            ::nntile::haul::stage::staged_submit_count();
        std::uint64_t const native_after =
            ::nntile::haul::gemm_native_chunked_count();
        INFO("staged submits before=" << staged_before << " after="
                                      << staged_after << " native gemm before="
                                      << native_before << " after="
                                      << native_after);
        REQUIRE(staged_after - staged_before == 0);
        REQUIRE(native_after - native_before >= 1);
    }

    auto const c_out = runtime.get_output<float>(tc);
    auto const t_out = runtime.get_output<float>(tt);
    auto const d_out = runtime.get_output<float>(td);
    auto const mse_out = runtime.get_output<float>(tmse);

    INFO("gemm+softmax max rel diff " << max_rel_diff(c_out, c_ref));
    REQUIRE(max_rel_diff(c_out, c_ref) < 1e-4f);
    INFO("maxsumexp max rel diff " << max_rel_diff(mse_out, mse_ref));
    REQUIRE(max_rel_diff(mse_out, mse_ref) < 1e-4f);
    INFO("transpose max rel diff " << max_rel_diff(t_out, t_ref));
    REQUIRE(max_rel_diff(t_out, t_ref) < 1e-4f);
    INFO("norm_slice max rel diff " << max_rel_diff(d_out, d_ref));
    REQUIRE(max_rel_diff(d_out, d_ref) < 1e-4f);
}

} // namespace

#ifdef NNTILE_USE_NNHAUL

namespace
{

// The embedding round-trip through the graph path on a vocab table well
// above the chunking threshold (16.4 MiB fp32: 4096 rows of 1000 elements,
// so rows deliberately straddle the default 4 MiB chunk class; the 8.4 MiB
// embed tile is chunked too, exercising the strided destination path of
// the forward and the whole-embed gather of the backward). Unique tokens
// (m*n = 2112 <= 4096) keep the backward single-contribution per element.
// Both ops must run their NATIVE chunked drivers: no staged submits.
void run_big_embedding_chunked_graph()
{
    Index const m = 33, n = 64, k = 1000, rows = 4096;
    std::size_t const vocab_nbytes = sizeof(float) * k * rows;
    REQUIRE(vocab_nbytes >= std::size_t{4} << 20);
    std::size_t const embed_n = static_cast<std::size_t>(m) * k * n;
    std::size_t const count = static_cast<std::size_t>(m) * n;

    bool const chunking_off = false; // one class: every tile is chunked
    if(chunking_off)
    {
        std::cout << "chunked_big embedding: NNHAUL_CHUNK_TILES=0 - "
                     "engagement assertions skipped, graph still runs\n";
    }

    std::vector<std::int64_t> idx(count);
    idx[0] = rows - 1; // last chunk
    idx[1] = rows - 2;
    for(std::size_t i = 2; i < count; ++i)
    {
        // rows 0..count-3 once each: row 0 (first chunk) and the rows
        // straddling the chunk boundaries (3, 6, 9, ...) included.
        idx[i] = static_cast<std::int64_t>(i) - 2;
    }
    std::vector<float> vocab(k * rows);
    for(std::size_t i = 0; i < vocab.size(); ++i)
    {
        vocab[i] = 0.01f * static_cast<float>(i % 9973) - 8.0f;
    }
    // Host reference: forward gather, then backward with beta=0.
    std::vector<float> embed_ref(embed_n, 0.0f);
    for(Index i2 = 0; i2 < n; ++i2)
    {
        for(Index i1 = 0; i1 < m; ++i1)
        {
            Index const token = idx[static_cast<std::size_t>(i2) * m + i1];
            for(Index i0 = 0; i0 < k; ++i0)
            {
                embed_ref[static_cast<std::size_t>((i2 * k + i0) * m + i1)] =
                    vocab[static_cast<std::size_t>(token) * k + i0];
            }
        }
    }
    std::vector<float> grad_ref(k * rows, 0.0f);
    for(Index i2 = 0; i2 < n; ++i2)
    {
        for(Index i1 = 0; i1 < m; ++i1)
        {
            Index const token = idx[static_cast<std::size_t>(i2) * m + i1];
            for(Index i0 = 0; i0 < k; ++i0)
            {
                float const e = embed_ref[static_cast<std::size_t>((i2 * k + i0) * m + i1)];
                grad_ref[static_cast<std::size_t>(token) * k + i0] +=
                    0.5f * e;
            }
        }
    }

    TileGraph graph("chunked_big_embedding");
    auto *tindex = graph.data({m, n}, "index", DataType::INT64);
    auto *tvocab = graph.data({k, rows}, "vocab", DataType::FP32);
    auto *tembed = graph.data({m, k, n}, "embed", DataType::FP32);
    auto *tgrad = graph.data({k, rows}, "grad", DataType::FP32);
    tg::embedding(m, n, k, Index{0}, k, tindex, tvocab, tembed);
    tg::embedding_backward(m, n, k, Index{0}, k, Scalar{0.5},
        Scalar{0.0}, tindex, tembed, tgrad, 0);

    std::uint64_t const staged_before =
        chunking_off ? 0 : ::nntile::haul::stage::staged_submit_count();
    Runtime runtime(graph);
    runtime.compile();
    runtime.bind_data(tindex, idx);
    runtime.bind_data(tvocab, vocab);
    runtime.bind_data(tembed, std::vector<float>(embed_n, 0.0f));
    runtime.bind_data(tgrad, std::vector<float>(k * rows, 0.0f));
    runtime.execute();
    runtime.wait();

    if(!chunking_off)
    {
        // Both ops are natively chunked: nothing may take the staging
        // fallback (the bind_data copies are the native copy op).
        std::uint64_t const staged_after =
            ::nntile::haul::stage::staged_submit_count();
        INFO("embedding round-trip staged submits before=" << staged_before
                                                           << " after="
                                                           << staged_after);
        REQUIRE(staged_after == staged_before);
    }

    auto const embed_out = runtime.get_output<float>(tembed);
    auto const grad_out = runtime.get_output<float>(tgrad);
    std::vector<fp32_t> const embed_ref_t(embed_ref.begin(), embed_ref.end());
    std::vector<fp32_t> const grad_ref_t(grad_ref.begin(), grad_ref.end());
    INFO("embedding fwd max rel diff "
         << max_rel_diff(embed_out, embed_ref_t));
    REQUIRE(max_rel_diff(embed_out, embed_ref_t) < 1e-6f);
    INFO("embedding backward max rel diff "
         << max_rel_diff(grad_out, grad_ref_t));
    REQUIRE(max_rel_diff(grad_out, grad_ref_t) < 1e-6f);
}

} // namespace

#endif // NNTILE_USE_NNHAUL

#ifdef NNTILE_USE_NNHAUL

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Chunked big-tile graph path (default ladder)",
    "[graph][tile][chunked]")
{
    run_big_tile_chunked_graph();
    // Second mode: a pristine child process under the deep ladder
    // (NNHAUL_CHUNK_SIZES=4096 NNHAUL_CHUNK_MIN_COUNT=1 -> 512 chunks
    // per 2 MiB tile). The knobs are parsed once per process by nnhaul
    // (function-local statics read at the first chunk_bytes_for call,
    // which happens after nnhaul::init), so the deep mode must re-exec.
    // /proc/self/exe inside the system() shell resolves to the shell,
    // so resolve the test binary's own path first.
    char self_buf[4096];
    ssize_t const self_len =
        ::readlink("/proc/self/exe", self_buf, sizeof(self_buf) - 1);
    REQUIRE(self_len > 0);
    self_buf[self_len] = '\0';
    std::string const self(self_buf);
    std::string const cmd = "NNHAUL_CHUNK_SIZES=4096 "
                            "NNHAUL_CHUNK_MIN_COUNT=1 '" +
                            self +
                            "' 'Chunked big-tile graph path (deep "
                            "chunks)'";
    int const rc = std::system(cmd.c_str());
    INFO("deep-mode child exit status " << rc);
    REQUIRE(rc == 0);
}

// Hidden tag: only the re-exec'ed child names this case explicitly; a
// plain run of the binary skips it (the parent process is pinned to the
// default ladder, where the deep-ladder precondition is false).
TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Chunked big-tile graph path (deep chunks)",
    "[.][graph][tile][chunked][deep]")
{
    // Reached in the child process (see the default-ladder case). The
    // ladder knobs were parsed from the environment this process was
    // exec'd with; the engagement assertions inside the routine verify
    // the deep chunk counts. Under NNHAUL_CHUNK_TILES=0 chunking is off
    // and the deep precondition is vacuous - the graph still runs.
    if ((2u << 20) <= ::nnhaul::kChunkClassBytes)
    {
        REQUIRE(::nnhaul::kChunkClassBytes >= 4096);
    }
    run_big_tile_chunked_graph();
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Chunked big-tile embedding round-trip",
    "[graph][tile][chunked]")
{
    run_big_embedding_chunked_graph();
}

#else

TEST_CASE("Chunked big-tile graph path requires NNHaul", "[chunked]")
{
    SUCCEED("StarPU build: chunked-tile path does not exist");
}

#endif
