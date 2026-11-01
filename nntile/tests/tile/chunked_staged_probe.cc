/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile/chunked_staged_probe.cc
 * Adversarial probes for the universal staging fallback on chunked
 * tiles: tight-cap staged gemm through the graph (a), a mixed
 * native-elementwise + staged chain compared bitwise against
 * contiguous mode (b), scratch-pressure fan-out (c), and the
 * compile-time torch-op gate (d).
 *
 * (RULE 7 VACUITY NOTE, 2026-10-18) this probe was written for the
 * configurable-class era and its tiles are 2-3 MiB -- every tile is a
 * SINGLE chunk under the compile-time 16 MiB class, so the probes that
 * keyed on chunk_bytes_for(2/3 MiB) > 0 no longer engage their chunked
 * intent: probe a's native-chunked-gemm assert is skipped (the guard
 * `chunking_off` is true), and probes b/e spawn their contiguous-mode
 * children WITHOUT the retired FORCE_PANEL_BYTES / CHUNK_UNITS_FORCE_
 * BYTES knobs (the env knobs are gone and chunk_bytes_for returns 0),
 * so those comparisons degenerate to plain flag-on vs flag-off
 * contiguous runs. The staged-fallback pressure probes (c) and the
 * torch gate (d) remain meaningful. The chunked-side coverage this
 * file once bought lives in the .zcode/bench-kernel checks
 * (p4_check BIG blocks, bv_check, ring_check) on REAL 16 MiB
 * factorizations; making these probes chunked again requires
 * >16 MiB tiles and a rewrite for the constant-class era (recorded on
 * the campaign page's coverage-lost list).
 *
 * Knobs (read per case, before the Context init): PROBE_CAP_MIB
 * (default unlimited), PROBE_POLICY (reactive|belady|proactive),
 * PROBE_REPS (default 3).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include <nntile/constants.hh>
#include <nntile/context.hh>
#include <nntile/core/tile.hh>
#include <nntile/kernel/cblas.hh>
#include <nntile/kernel/maxsumexp.hh>
#include <nntile/kernel/softmax_inplace.hh>
#include <nntile/kernel/swap_two_axes.hh>
#include <nntile/nnhaul/ops/gemm.hh>
#include <nntile/nnhaul/stage.hh>
#include <nntile/tile.hh>
#include <nntile/tile/ops/add_inplace.hh>
#include <nntile/tile/ops/fill.hh>
#include <nntile/tile/ops/gemm.hh>
#include <nntile/tile/ops/maxsumexp.hh>
#include <nntile/tile/ops/softmax_inplace.hh>
#include <nntile/tile/ops/swap_two_axes.hh>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace nntile;
namespace tg = nntile::tile;

namespace
{

std::size_t env_cap_bytes()
{
    char const *e = std::getenv("PROBE_CAP_MIB");
    if (e == nullptr)
    {
        return 0;
    }
    return static_cast<std::size_t>(std::atol(e)) << 20;
}

std::string env_policy()
{
    char const *e = std::getenv("PROBE_POLICY");
    return e == nullptr ? std::string("reactive") : std::string(e);
}

int env_reps()
{
    char const *e = std::getenv("PROBE_REPS");
    return e == nullptr ? 3 : std::max(1, std::atoi(e));
}

std::unique_ptr<nntile::Context> make_context()
{
    if (!env_policy().empty())
    {
        ::setenv("NNHAUL_OFFLOAD_POLICY", env_policy().c_str(), 1);
    }
    return std::make_unique<nntile::Context>(
        1, 0, 0, "/tmp/nntile_ooc", 16777216, 0, "localhost", 5001, 0, 0,
        env_cap_bytes());
}

float max_rel_diff(std::vector<float> const &got, std::vector<float> const &ref)
{
    float worst = 0.0f;
    for (std::size_t i = 0; i < got.size(); ++i)
    {
        float const scale =
            std::max(std::max(std::abs(ref[i]), std::abs(got[i])), 1.0f);
        worst = std::max(worst, std::abs(got[i] - ref[i]) / scale);
    }
    return worst;
}

std::uint64_t fnv1a(std::vector<float> const &data)
{
    std::uint64_t hash = 1469598103934665603ull;
    unsigned char const *bytes =
        reinterpret_cast<unsigned char const *>(data.data());
    for (std::size_t i = 0; i < data.size() * sizeof(float); ++i)
    {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

// (a) One chunked gemm through the graph on 2 MiB tiles: A 2 MiB, B 1
// MiB, C 2 MiB (5 MiB of payloads; under PROBE_CAP_MIB the runtime must
// offload/evict to make room). gemm runs the NATIVE chunked panel
// driver (no staging): the engagement assert is the native-driver
// counter, not the staged counter.
float run_staged_gemm_graph(Index m, Index n, Index k)
{
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
    std::vector<fp32_t> c_ref(static_cast<std::size_t>(m * n),
        fp32_t(0.0f));
#ifdef NNTILE_USE_CBLAS
    nntile::kernel::cblas::gemm<fp32_t>(
        nntile::TransOp(nntile::TransOp::NoTrans),
        nntile::TransOp(nntile::TransOp::NoTrans),
        m,
        n,
        k,
        Index{1},
        Scalar{1.0},
        af.data(),
        bf.data(),
        Scalar{0.0},
        c_ref.data());
#else
    // The CUDA-only NNHaul build links no CBLAS: a naive reference
    // serves the same purpose. Row-major throughout, matching the
    // tile shapes: A is m by k, B is k by n, C is m by n.
    for (Index i = 0; i < m; ++i)
    {
        for (Index j = 0; j < n; ++j)
        {
            float acc = 0.0f;
            for (Index l = 0; l < k; ++l)
            {
                acc += a[static_cast<std::size_t>(i * k + l)] *
                    b[static_cast<std::size_t>(l * n + j)];
            }
            c_ref[static_cast<std::size_t>(i * n + j)] = acc;
        }
    }
#endif

    TileGraph graph("probe_gemm");
    auto *ta = graph.data({m, k}, "a", DataType::FP32);
    auto *tb = graph.data({k, n}, "b", DataType::FP32);
    auto *tc = graph.data({m, n}, "c", DataType::FP32);
    tg::gemm(ta, tb, tc, Scalar{1.0}, Scalar{0.0}, false, false, Index{1},
        Index{0});
    Runtime runtime(graph);
    runtime.compile();
    runtime.bind_data(ta, a);
    runtime.bind_data(tb, b);
    std::vector<float> c_init(static_cast<std::size_t>(m * n), 0.0f);
    runtime.bind_data(tc, c_init);
    runtime.execute();
    runtime.wait();
    auto const c_out = runtime.get_output<float>(tc);
    return max_rel_diff(c_out,
        std::vector<float>(c_ref.begin(), c_ref.end()));
}

} // namespace

#ifdef NNTILE_USE_NNHAUL

// (a) Tight-cap native chunked gemm through the graph, several reps.
// Run once per offload policy: PROBE_POLICY=<reactive|belady|proactive>.
TEST_CASE("probe a: native chunked gemm under tight cap", "[chunked][probe]")
{
    auto context = make_context();
    // Class-relative geometry: k scaled so the m*k operand exceeds the
    // compile-time chunk class by 2x (multi-chunk under any class; the
    // CI default is 4 MiB and the old fixed 512 sized tiles below it).
    Index const m = 1024, n = 512;
    Index const k = static_cast<Index>(
        (2 * ::nnhaul::kChunkClassBytes / sizeof(float)) / m);
    bool const chunking_off =
        false; // one class: every tile is chunked
    std::uint64_t const native_before =
        ::nntile::haul::gemm_native_chunked_count();
    std::uint64_t const staged_before =
        ::nntile::haul::stage::staged_submit_count();
    for (int rep = 0; rep < env_reps(); ++rep)
    {
        float const diff = run_staged_gemm_graph(m, n, k);
        INFO("rep " << rep << " policy " << env_policy() << " cap "
                    << env_cap_bytes() << " max rel diff " << diff);
        REQUIRE(diff < 1e-4f);
    }
    if (!chunking_off)
    {
        // Chunked tiles: the gemm must be native (staging static).
        REQUIRE(::nntile::haul::gemm_native_chunked_count() > native_before);
    }
    REQUIRE(::nntile::haul::stage::staged_submit_count() == staged_before);
    std::cout << "probe a OK policy=" << env_policy()
              << " cap_mib=" << (env_cap_bytes() >> 20) << "\n";
}

namespace
{

// (b) The mixed chain fill -> gemm -> maxsumexp -> softmax_inplace ->
// add on shared 2 MiB tiles. Prints one RESULT line; the chunked run is
// compared bitwise against a contiguous-mode child
// (NNHAUL_CHUNK_TILES=0) running the identical chain.
std::uint64_t run_mixed_chain(bool print)
{
    Index const m = 1024, n = 512, k = 512;
    std::vector<float> a(static_cast<std::size_t>(m * k));
    std::vector<float> b(static_cast<std::size_t>(n * k));
    for (Index i = 0; i < m * k; ++i)
    {
        a[static_cast<std::size_t>(i)] =
            0.01f * static_cast<float>((i % 11) - 5);
    }
    for (Index i = 0; i < n * k; ++i)
    {
        b[static_cast<std::size_t>(i)] =
            0.01f * static_cast<float>((i % 7) - 3);
    }
    TileGraph graph("probe_chain");
    auto *ta = graph.data({m, k}, "a", DataType::FP32);
    auto *tb = graph.data({k, n}, "b", DataType::FP32);
    auto *tc = graph.data({m, n}, "c", DataType::FP32);
    auto *tmse = graph.data({n, 2}, "mse", DataType::FP32);
    auto *ty = graph.data({m, n}, "y", DataType::FP32);
    tg::fill(Scalar{0.25}, ta); // native chunk-loop elementwise
    tg::fill(Scalar{0.5}, tb);  // native
    tg::gemm(ta, tb, tc, Scalar{1.0}, Scalar{0.0}, false, false, Index{1},
        Index{0}); // native chunked panel driver
    tg::maxsumexp(tc, tmse, Index{0}, Scalar{0.0}, 0); // native chunked
    tg::softmax_inplace(tmse, Scalar{1.0}, tc, Index{0}); // native chunked
    tg::add_inplace(Scalar{2.0}, tc, Scalar{-0.5}, ty); // native elementwise
    Runtime runtime(graph);
    runtime.compile();
    std::vector<float> a_init(a.size(), 1.0f);
    runtime.bind_data(ta, a_init);
    runtime.bind_data(tb, b);
    std::vector<float> c_init(static_cast<std::size_t>(m * n), 0.0f);
    runtime.bind_data(tc, c_init);
    std::vector<float> mse_init(static_cast<std::size_t>(2 * n), 0.0f);
    runtime.bind_data(tmse, mse_init);
    std::vector<float> y_init(static_cast<std::size_t>(m * n), 7.0f);
    runtime.bind_data(ty, y_init);
    runtime.execute();
    runtime.wait();
    auto const y_out = runtime.get_output<float>(ty);
    std::uint64_t const hash = fnv1a(y_out);
    if (print)
    {
        std::printf("RESULT %016llx\n",
            static_cast<unsigned long long>(hash));
    }
    return hash;
}

} // namespace

TEST_CASE("probe b: mixed chain bitwise vs contiguous mode",
    "[chunked][probe]")
{
    // RETIRED PREMISE: the child-side knob NNHAUL_CHUNK_TILES=0 is
    // dead — the plan engine always uses the compile-time chunk class
    // and prints a retirement warning where the child's RESULT parsing
    // looked, so the "contiguous mode" half of the comparison no
    // longer exists. Chunked-vs-classic parity is enforced in-process
    // by chunked_ab (and probe a's reference check). Skipped, not red.
    SKIP("NNHAUL_CHUNK_TILES=0 is retired: no contiguous-mode child "
         "exists under the plan engine");
}
#if 0 // retired child-mode body, kept for reference
    char self_buf[4096];
    ssize_t const self_len =
        ::readlink("/proc/self/exe", self_buf, sizeof(self_buf) - 1);
    REQUIRE(self_len > 0);
    self_buf[self_len] = '\0';
    // The child runs unchunked tiles, but its gemm must be factored the
    // SAME way as this chunked run's native panel driver, or cuBLAS's
    // different call shapes reorder the fp32 sums (last-ulp drift that
    // the downstream softmax would amplify). NNHAUL_GEMM_FORCE_PANEL_BYTES
    // makes an unchunked gemm body use virtual panels of exactly the
    // chunk class this process's tiles got (all gemm operands here fall
    // into the same 2 MiB-tile class). After the family-1 promotion the
    // same holds for the chain's unit-structured ops: maxsumexp and
    // softmax_inplace run native chunked unit drivers whose piecewise
    // beta-chain reorders the reduction sums, so the child must force
    // the identical virtual chunking with NNHAUL_CHUNK_UNITS_FORCE_BYTES
    // to issue the same sub-call sequence (softmax_inplace is elementwise
    // and would match bitwise either way; maxsumexp would not).
    std::size_t const panel_bytes =
        ::nnhaul::kChunkClassBytes;
    std::string const cmd =
        std::string("NNHAUL_CHUNK_TILES=0") +
        (panel_bytes > 0
                ? " NNHAUL_GEMM_FORCE_PANEL_BYTES=" +
                      std::to_string(panel_bytes)
                      + " NNHAUL_CHUNK_UNITS_FORCE_BYTES=" +
                      std::to_string(panel_bytes)
                : std::string()) +
        " '" + std::string(self_buf) +
        "' 'probe b: mixed chain bitwise vs "
        "contiguous mode'";
    std::FILE *pipe = ::popen(cmd.c_str(), "r");
    REQUIRE(pipe != nullptr);
    std::uint64_t contiguous_hash = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), pipe) != nullptr)
    {
        std::cout << line;
        if (std::strncmp(line, "RESULT ", 7) == 0)
        {
            contiguous_hash = std::strtoull(line + 7, nullptr, 16);
        }
    }
    int const rc = ::pclose(pipe);
    INFO("chunked " << std::hex << hash << " contiguous "
                    << contiguous_hash << std::dec);
    REQUIRE(rc == 0);
    REQUIRE(contiguous_hash != 0);
    REQUIRE(hash == contiguous_hash);
    std::cout << "probe b OK bitwise\n";
#endif // retired

// (c) Chunked fan-out under scratch pressure: many distinct 2 MiB
// chunked swap_two_axes ops under a tiny cap. swap_two_axes (like
// transpose and every other op except gemm) is a NATIVE chunk-addressed
// kernel — the historical V1 staging fallback is gone — so this probe
// now doubles as the law's enforcement: the whole-buffer staging
// counter must not move (only gemm's workspace staging is sanctioned,
// and it does not go through stage::staged_insert_task).
TEST_CASE("probe c: ten chunked ops under scratch pressure",
    "[chunked][probe]")
{
    auto context = make_context();
    Index const m = 512, n = 1024; // 2 MiB fp32 tiles
    char const *ce = std::getenv("PROBE_COUNT");
    int const count = ce == nullptr ? 10 : std::max(1, std::atoi(ce));
    std::uint64_t const staged_before =
        ::nntile::haul::stage::staged_submit_count();
    std::vector<std::vector<float>> srcs(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        srcs[static_cast<std::size_t>(i)].resize(
            static_cast<std::size_t>(m * n));
        for (Index j = 0; j < m * n; ++j)
        {
            srcs[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                static_cast<float>(i) + 0.001f * static_cast<float>(j % 97);
        }
    }
    TileGraph graph("probe_fanout");
    std::vector<TileGraph::TileNode *> src_nodes, dst_nodes;
    for (int i = 0; i < count; ++i)
    {
        src_nodes.push_back(
            graph.data({m, n}, "s" + std::to_string(i), DataType::FP32));
        dst_nodes.push_back(
            graph.data({n, m}, "d" + std::to_string(i), DataType::FP32));
        tg::swap_two_axes(src_nodes.back(), dst_nodes.back(), Index{0},
            Index{1});
    }
    Runtime runtime(graph);
    runtime.compile();
    for (int i = 0; i < count; ++i)
    {
        runtime.bind_data(src_nodes[static_cast<std::size_t>(i)],
            srcs[static_cast<std::size_t>(i)]);
        std::vector<float> dst_init(
            static_cast<std::size_t>(m * n), -1.0f);
        runtime.bind_data(dst_nodes[static_cast<std::size_t>(i)], dst_init);
    }
    runtime.execute();
    runtime.wait();
    for (int i = 0; i < count; ++i)
    {
        auto const out =
            runtime.get_output<float>(dst_nodes[static_cast<std::size_t>(i)]);
        std::vector<float> const &src =
            srcs[static_cast<std::size_t>(i)];
        std::vector<fp32_t> srcf(src.begin(), src.end());
        std::vector<fp32_t> ref(static_cast<std::size_t>(m * n));
        nntile::kernel::swap_two_axes::cpu<fp32_t>(1, m, 1, n, 1,
            srcf.data(), ref.data());
        // The 2-D axis swap decomposes to the 5-D kernel shape
        // (1, m, 1, n, 1) - see core/swap_two_axes.cc. Bitwise: the
        // swap only moves bytes.
        float const diff = max_rel_diff(out,
            std::vector<float>(ref.begin(), ref.end()));
        REQUIRE(diff == 0.0f);
    }
    // The law (only gemm stages, through its workspace): the stage
    // fallback counter must be exactly where it started.
    REQUIRE(::nntile::haul::stage::staged_submit_count() ==
        staged_before);
    std::cout << "probe c OK staged_submits="
              << ::nntile::haul::stage::staged_submit_count() << "\n";
}

namespace
{

// (e) A graph of ONLY gemms - the GPT-2-shaped workload: weights
// {768,3072} (9 MiB fp32), activations {1024,768} (3 MiB), four chained
// gemms alternating NoTrans/Trans use of the weights. On chunked tiles
// every gemm takes the native panel driver; prints one RESULT line for
// the bitwise comparison against the contiguous-mode child.
[[maybe_unused]] std::uint64_t run_gemm_chain(bool print)
{
    Index const m = 1024, d = 768, h = 3072;
    std::vector<float> w(static_cast<std::size_t>(d * h));
    std::vector<float> x(static_cast<std::size_t>(m * d));
    for (Index i = 0; i < d * h; ++i)
    {
        w[static_cast<std::size_t>(i)] =
            0.01f * static_cast<float>((i % 19) - 9);
    }
    for (Index i = 0; i < m * d; ++i)
    {
        x[static_cast<std::size_t>(i)] =
            0.01f * static_cast<float>((i % 23) - 11);
    }
    TileGraph graph("probe_gemm_chain");
    auto *tw = graph.data({d, h}, "w", DataType::FP32);
    auto *tx = graph.data({m, d}, "x", DataType::FP32);
    auto *th1 = graph.data({m, h}, "h1", DataType::FP32);
    auto *th2 = graph.data({m, d}, "h2", DataType::FP32);
    auto *th3 = graph.data({m, h}, "h3", DataType::FP32);
    auto *tout = graph.data({m, d}, "out", DataType::FP32);
    tg::gemm(tx, tw, th1, Scalar{1.0}, Scalar{0.0}, false, false, Index{1},
        Index{0}); // x * w: NoTrans,NoTrans
    tg::gemm(th1, tw, th2, Scalar{1.0}, Scalar{0.0}, false, true, Index{1},
        Index{0}); // h1 * w^T: NoTrans,Trans
    tg::gemm(th2, tw, th3, Scalar{1.0}, Scalar{0.0}, false, false, Index{1},
        Index{0});
    tg::gemm(th3, tw, tout, Scalar{1.0}, Scalar{0.0}, false, true, Index{1},
        Index{0});
    Runtime runtime(graph);
    runtime.compile();
    runtime.bind_data(tw, w);
    runtime.bind_data(tx, x);
    std::vector<float> zero_h(static_cast<std::size_t>(m * h), 0.0f);
    std::vector<float> zero_d(static_cast<std::size_t>(m * d), 0.0f);
    runtime.bind_data(th1, zero_h);
    runtime.bind_data(th2, zero_d);
    runtime.bind_data(th3, zero_h);
    runtime.bind_data(tout, zero_d);
    runtime.execute();
    runtime.wait();
    auto const out = runtime.get_output<float>(tout);
    std::uint64_t const hash = fnv1a(out);
    if (print)
    {
        std::printf("RESULT %016llx\n",
            static_cast<unsigned long long>(hash));
    }
    return hash;
}

} // namespace

// (e) The gemm-only chain on chunked tiles, one session per offload
// policy (reactive, belady, proactive), each bitwise against a
// contiguous-mode child whose gemm bodies are panel-forced to this
// process's chunk class (NNHAUL_GEMM_FORCE_PANEL_BYTES) so the child
// issues the identical cublas call factorization. The gemms must be
// NATIVE (staging counter static, native counter +4 per chain). A cap
// may be applied with PROBE_CAP_MIB, but it must admit one gemm's full
// operand set (the frontier fetches every chunk before launch): this
// graph's largest working set is x/w/h1 = 24 MiB.
TEST_CASE("probe e: gemm-only chain bitwise vs contiguous mode",
    "[chunked][probe]")
{
    // RETIRED PREMISE (same as probe b): NNHAUL_CHUNK_TILES=0 is dead,
    // no contiguous-mode child exists. The gemm-native/staged counter
    // assertions live on in probe a; skipped, not red.
    SKIP("NNHAUL_CHUNK_TILES=0 is retired: no contiguous-mode child "
         "exists under the plan engine");
#if 0 // retired child-mode body, kept for reference
    char const *mode = ::getenv("NNHAUL_CHUNK_TILES");
    bool const is_child = mode != nullptr && std::strcmp(mode, "0") == 0;
    if (is_child)
    {
        auto context = make_context();
        run_gemm_chain(true);
        return;
    }
    char self_buf[4096];
    ssize_t const self_len =
        ::readlink("/proc/self/exe", self_buf, sizeof(self_buf) - 1);
    REQUIRE(self_len > 0);
    self_buf[self_len] = '\0';
    std::size_t const panel_bytes =
        ::nnhaul::kChunkClassBytes;
    std::string const force = panel_bytes > 0
        ? " NNHAUL_GEMM_FORCE_PANEL_BYTES=" + std::to_string(panel_bytes)
        : std::string();
    for (char const *policy : {"reactive", "belady", "proactive"})
    {
        ::setenv("PROBE_POLICY", policy, 1);
        ::setenv("NNHAUL_OFFLOAD_POLICY", policy, 1);
        auto context = make_context();
        std::uint64_t const native_before =
            ::nntile::haul::gemm_native_chunked_count();
        std::uint64_t const staged_before =
            ::nntile::haul::stage::staged_submit_count();
        std::uint64_t const hash = run_gemm_chain(false);
        INFO("policy " << policy << " hash " << std::hex << hash << std::dec);
        if (panel_bytes > 0)
        {
            REQUIRE(::nntile::haul::gemm_native_chunked_count()
                - native_before >= 4);
            REQUIRE(::nntile::haul::stage::staged_submit_count()
                == staged_before);
        }
        std::string const cmd = "PROBE_POLICY=" + std::string(policy) +
            " NNHAUL_OFFLOAD_POLICY=" + std::string(policy) +
            " NNHAUL_CHUNK_TILES=0" + force + " '" + std::string(self_buf) +
            "' 'probe e: gemm-only chain bitwise vs contiguous mode'";
        std::FILE *pipe = ::popen(cmd.c_str(), "r");
        REQUIRE(pipe != nullptr);
        std::uint64_t contiguous_hash = 0;
        char line[256];
        while (std::fgets(line, sizeof(line), pipe) != nullptr)
        {
            if (std::strncmp(line, "RESULT ", 7) == 0)
            {
                contiguous_hash = std::strtoull(line + 7, nullptr, 16);
            }
        }
        int const rc = ::pclose(pipe);
        INFO("policy " << policy << " chunked " << std::hex << hash
                       << " contiguous " << contiguous_hash << std::dec);
        REQUIRE(rc == 0);
        REQUIRE(contiguous_hash != 0);
        REQUIRE(hash == contiguous_hash);
        std::cout << "probe e OK policy=" << policy << "\n";
    }
#endif // retired
}

namespace
{

// A stand-in torch_dispatch op for the compile-time gate: nothing torch
// native is linked into this build, and the gate matches the
// "TILE_TORCH_" op-name prefix.
struct FakeTorchOp : TileGraph::OpNode
{
    TileGraph::TileNode *in = nullptr;
    TileGraph::TileNode *out = nullptr;

    FakeTorchOp(TileGraph::TileNode *in_, TileGraph::TileNode *out_)
        : in(in_), out(out_)
    {
        inputs_ = {in};
        outputs_ = {out};
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_PROBE";
    }

    void execute(Runtime &) const override {}

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<FakeTorchOp>(*this);
    }
};

} // namespace

// (d) A graph that lowers a torch_dispatch op while chunked tile
// handles are alive must fail at COMPILE time with the loud
// NNHAUL_CHUNK_TILES=0 message.
TEST_CASE("probe d: torch op plus chunked tiles throws at compile",
    "[chunked][probe]")
{
    auto context = make_context();
    // Live chunked tiles: a classic graph over 2 MiB tiles, compiled so
    // allocate_missing_tiles materializes chunked payloads (kept alive
    // by the graph).
    TileGraph classic("probe_classic");
    auto *big = classic.data({1024, 512}, "big", DataType::FP32);
    auto *big2 = classic.data({512, 1024}, "big2", DataType::FP32);
    tg::transpose(Scalar{1.0}, big, big2, Index{1});
    {
        Runtime runtime(classic);
        runtime.compile();
    }
    if (false) // one class: every tile is chunked
    {
        // NNHAUL_CHUNK_TILES=0: chunked tiles cannot exist, the
        // forbidden combination is unreachable by construction.
        SUCCEED("chunking disabled: gate combination unreachable");
        return;
    }
    REQUIRE(::nntile::haul::stage::chunked_handle_count() > 0);
    // The forbidden combination: a torch op in the same session.
    TileGraph torchy("probe_torchy");
    auto *x = torchy.data({4}, "x", DataType::FP32);
    auto *y = torchy.data({4}, "y", DataType::FP32);
    torchy.add_op(std::make_shared<FakeTorchOp>(x, y));
    Runtime runtime(torchy);
    bool threw = false;
    std::string message;
    try
    {
        runtime.compile();
    }
    catch (std::runtime_error const &err)
    {
        threw = true;
        message = err.what();
    }
    INFO(message);
    REQUIRE(threw);
    REQUIRE(message.find("NNHAUL_CHUNK_TILES=0") != std::string::npos);
    // Loud, not silent: nothing was demoted or executed.
    std::cout << "probe d OK: " << message << "\n";
}

#else

TEST_CASE("chunked staged probes require NNHaul", "[chunked][probe]")
{
    SUCCEED("StarPU build: chunked-tile path does not exist");
}

#endif
