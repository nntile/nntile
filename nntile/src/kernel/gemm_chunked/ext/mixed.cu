/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm_chunked/ext/mixed.cu
 * MIXED layouts of the GPT-2 MLP GEMMs: one or two operands are
 * single-chunk (one flat allocation), the rest are chunk tables. The
 * family's plain bridge (id 40) exists only with both A and B flat or
 * both table-addressed, so `ftt`, `tft`, `ftf`, `tff` -- and all-flat
 * operands under a table C (`fft`) -- were declined. This piece runs
 * them on the SIMT fp32 mainloop of mixed_bridge.cuh: the addressing is
 * chosen per operand, a flat operand may go through CUTLASS's own
 * iterators, a flat C through CUTLASS's own epilogue, and the
 * underfilled grids get a deterministic K split (single-owner tiles,
 * then the k pieces to a scratch, then a fixed-order reduction).
 *
 * The piece claims the measured GPT-2 small, medium and large MLP
 * shapes by exact shape and layout; other calls keep their routing.
 *
 * @version 1.0.0
 * */

// Corresponding header
#include "../gemm_chunked_ext.hh"

#ifdef NNTILE_USE_CUDA

// Third-party headers: the bridge header relies on these (as cuda.cu
// includes them); CUTLASS 2.11 is on this directory's include path.
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>

// The piece's kernels
#include "mixed_bridge.cuh"

#include <algorithm>
#include <cstdio>

namespace nntile::kernel::gemm_chunked
{

namespace ext
{

namespace
{

namespace md = mixed_detail;

//! A configuration id of this piece is an index into MIXED_VARIANTS, the
//! list of the kernels this translation unit builds. The dispatcher
//! adds its base before handing an id out, so the list holds fewer than
//! 100 entries.

//! The kernels this piece builds, one per line: (name, tile, native
//! loaders, native epilogue, occupancy hint, AC_A, AC_B, combo). A name
//! reads <tile><loaders><epilogue><ctas>_<A><B>_<op>:
//!   tile      d = 128x128 (256 threads), h = 128x64, v = 64x128
//!             (128 threads; half-size tiles fill a ragged grid better)
//!   loaders   q = the quad loaders for both operands,
//!             n = CUTLASS's own iterators for the flat operands
//!   epilogue  c = the table-aware iterator (C flat or table at runtime),
//!             f = CUTLASS's own iterator (flat C)
//!   ctas      1 = no hint, 2 = two CTAs resident per SM (launch bounds)
//! and <A><B> is the (flat, table) addressing pair, <op> the transposes.
//! Every line serves a claim below (the measured winner of its case).
#define MIXED_VARIANTS(X)                                                     \
    /* 128x128, quad loaders, table-aware epilogue */                         \
    X(dqc1_ft_nn, 0, 0, 0, 0, 0, 1, 0)                                        \
    X(dqc1_ft_tn, 0, 0, 0, 0, 0, 1, 2)                                        \
    X(dqc2_ff_nn, 0, 0, 0, 1, 0, 0, 0)                                        \
    X(dqc2_ff_nt, 0, 0, 0, 1, 0, 0, 1)                                        \
    X(dqc2_tt_nt, 0, 0, 0, 1, 1, 1, 1)                                        \
    /* 128x128, CUTLASS iterators for the flat operand */                     \
    X(dnc1_ft_nn, 0, 1, 0, 0, 0, 1, 0)                                        \
    X(dnc1_ft_nt, 0, 1, 0, 0, 0, 1, 1)                                        \
    X(dnc1_ff_nt, 0, 1, 0, 0, 0, 0, 1)                                        \
    X(dnc1_tf_tn, 0, 1, 0, 0, 1, 0, 2)                                        \
    X(dnc2_ft_nt, 0, 1, 0, 1, 0, 1, 1)                                        \
    X(dnc2_ft_tn, 0, 1, 0, 1, 0, 1, 2)                                        \
    X(dnc2_tf_tn, 0, 1, 0, 1, 1, 0, 2)                                        \
    /* 128x128, flat-C epilogue (C is flat) */                                \
    X(dqf1_tt_nn, 0, 0, 1, 0, 1, 1, 0)                                        \
    X(dqf1_tt_nt, 0, 0, 1, 0, 1, 1, 1)                                        \
    X(dqf1_ft_tn, 0, 0, 1, 0, 0, 1, 2)                                        \
    X(dqf1_tf_nn, 0, 0, 1, 0, 1, 0, 0)                                        \
    X(dnf1_tf_nt, 0, 1, 1, 0, 1, 0, 1)                                        \
    X(dnf1_tf_tn, 0, 1, 1, 0, 1, 0, 2)                                        \
    /* 128x64 (h) and 64x128 (v) tiles, 128 threads */                        \
    X(hq1_tf_tn, 1, 0, 0, 0, 1, 0, 2)                                         \
    X(hq1_tt_nt, 1, 0, 0, 0, 1, 1, 1)                                         \
    X(vq1_ft_tn, 2, 0, 0, 0, 0, 1, 2)                                         \
    X(vq1_tt_nn, 2, 0, 0, 0, 1, 1, 0)                                         \
    X(vn1_ff_nn, 2, 1, 0, 0, 0, 0, 0)

#define MIXED_ENUM_ID(nm, tile, nat, natc, occ, aca, acb, combo) nm,
enum MixedVariantId : int
{
    MIXED_VARIANTS(MIXED_ENUM_ID) kNumVariants
};
#undef MIXED_ENUM_ID

//! Addressing of an operand: 0 flat (shift 0), 1 table (any other shift,
//! the one-shot sentinel included).
constexpr int ac_of(int shift) noexcept { return shift == 0 ? 0 : 1; }

constexpr unsigned char kNN = 0, kNT = 1, kTN = 2;

//! One exact claim: a GPT-2 MLP GEMM (shape, transposes) on one layout,
//! with the kernel that serves it and its split schedule.
struct Claim
{
    int m;
    int n;
    int k;
    unsigned char combo;  //!< 2 * transA + transB
    unsigned char layout; //!< bit 0 A table, bit 1 B table, bit 2 C table
    int cfg;              //!< this piece's configuration id (a variant)
    int full;             //!< split: output tiles with a single owner
    int slices;           //!< split: k pieces per remaining tile (1: none)
};

//! One claim row: the layout is three letters, f flat / t table, for A, B
//! and C; VARIANT is a MIXED_VARIANTS name; the schedule is (FULL,
//! SLICES).
#define MIXED_ROW(M, N, K, COMBO, LA, LB, LC, VARIANT, FULL, SLICES)          \
    {                                                                         \
        M, N, K, COMBO,                                                       \
            static_cast<unsigned char>(                                       \
                ((LA) == 't') | (((LB) == 't') << 1) | (((LC) == 't') << 2)), \
            VARIANT, FULL, SLICES                                             \
    }

//! The measured claims: GPT-2 small / medium / large, 1024 and 4096
//! tokens, the six MLP GEMMs, on the layouts the 4 MiB and the 16 MiB
//! engines give them (an operand that fits one chunk is flat). The
//! schedule (FULL, SLICES) is the split of an underfilled grid: the
//! first FULL output tiles keep one owner, each of the rest is cut
//! into SLICES k pieces.
constexpr Claim const kClaims[] = {
    // ---- 4 MiB, GPT-2 medium, 1024 tokens ----
    MIXED_ROW(1024,
        4096,
        1024,
        kNN,
        'f',
        't',
        't',
        dqc1_ft_nn,
        0,
        1), // mlp_fc_dgrad
    MIXED_ROW(
        1024, 4096, 1024, kNT, 'f', 't', 't', dnc1_ft_nt, 0, 1), // mlp_fc_fwd
    MIXED_ROW(
        1024, 4096, 1024, kTN, 'f', 't', 't', vq1_ft_tn, 0, 1), // mlp_fc_wgrad
    MIXED_ROW(1024,
        1024,
        4096,
        kNN,
        't',
        't',
        'f',
        dqf1_tt_nn,
        0,
        1), // mlp_proj_dgrad
    MIXED_ROW(1024,
        1024,
        4096,
        kNT,
        't',
        't',
        'f',
        dqc2_tt_nt,
        0,
        1), // mlp_proj_fwd
    MIXED_ROW(4096,
        1024,
        1024,
        kTN,
        't',
        'f',
        't',
        hq1_tf_tn,
        0,
        1), // mlp_proj_wgrad
    // ---- 4 MiB, GPT-2 small, 1024 tokens ----
    MIXED_ROW(
        1024, 3072, 768, kNN, 'f', 't', 't', dqc1_ft_nn, 0, 1), // mlp_fc_dgrad
    MIXED_ROW(
        1024, 3072, 768, kNT, 'f', 't', 't', dnc1_ft_nt, 0, 1), // mlp_fc_fwd
    MIXED_ROW(768,
        3072,
        1024,
        kTN,
        'f',
        't',
        't',
        dnc2_ft_tn,
        136,
        8), // mlp_fc_wgrad
    MIXED_ROW(1024,
        768,
        3072,
        kNN,
        't',
        't',
        'f',
        vq1_tt_nn,
        0,
        2), // mlp_proj_dgrad
    MIXED_ROW(
        1024, 768, 3072, kNT, 't', 't', 'f', hq1_tt_nt, 0, 2), // mlp_proj_fwd
    MIXED_ROW(3072,
        768,
        1024,
        kTN,
        't',
        'f',
        't',
        dnc2_tf_tn,
        136,
        8), // mlp_proj_wgrad
    // ---- 16 MiB, GPT-2 large, 1024 tokens ----
    MIXED_ROW(1024,
        5120,
        1280,
        kNN,
        'f',
        't',
        't',
        dnc1_ft_nn,
        272,
        4), // mlp_fc_dgrad
    MIXED_ROW(
        1024, 5120, 1280, kNT, 'f', 't', 't', dnc2_ft_nt, 0, 1), // mlp_fc_fwd
    MIXED_ROW(
        1280, 5120, 1024, kTN, 'f', 't', 't', vq1_ft_tn, 0, 1), // mlp_fc_wgrad
    MIXED_ROW(1024,
        1280,
        5120,
        kNN,
        't',
        't',
        'f',
        dqf1_tt_nn,
        68,
        5), // mlp_proj_dgrad
    MIXED_ROW(1024,
        1280,
        5120,
        kNT,
        't',
        't',
        'f',
        dqf1_tt_nt,
        68,
        5), // mlp_proj_fwd
    MIXED_ROW(5120,
        1280,
        1024,
        kTN,
        't',
        'f',
        't',
        dnc2_tf_tn,
        0,
        1), // mlp_proj_wgrad
    // ---- 16 MiB, GPT-2 medium, 4096 tokens ----
    MIXED_ROW(4096,
        4096,
        1024,
        kNN,
        'f',
        'f',
        't',
        dqc2_ff_nn,
        1020,
        17), // mlp_fc_dgrad
    MIXED_ROW(4096,
        4096,
        1024,
        kNT,
        'f',
        'f',
        't',
        dqc2_ff_nt,
        1020,
        17), // mlp_fc_fwd
    MIXED_ROW(1024,
        4096,
        4096,
        kTN,
        'f',
        't',
        'f',
        dqf1_ft_tn,
        0,
        1), // mlp_fc_wgrad
    MIXED_ROW(4096,
        1024,
        4096,
        kNN,
        't',
        'f',
        'f',
        dqf1_tf_nn,
        0,
        1), // mlp_proj_dgrad
    MIXED_ROW(4096,
        1024,
        4096,
        kNT,
        't',
        'f',
        'f',
        dnf1_tf_nt,
        0,
        1), // mlp_proj_fwd
    MIXED_ROW(4096,
        1024,
        4096,
        kTN,
        't',
        'f',
        'f',
        dnf1_tf_tn,
        0,
        1), // mlp_proj_wgrad
    // ---- 16 MiB, GPT-2 small, 4096 tokens ----
    MIXED_ROW(
        4096, 3072, 768, kNN, 'f', 'f', 't', vn1_ff_nn, 0, 1), // mlp_fc_dgrad
    MIXED_ROW(
        4096, 3072, 768, kNT, 'f', 'f', 't', dnc1_ff_nt, 0, 1), // mlp_fc_fwd
    MIXED_ROW(768,
        3072,
        4096,
        kTN,
        'f',
        't',
        'f',
        dqc1_ft_tn,
        136,
        8), // mlp_fc_wgrad
    MIXED_ROW(4096,
        768,
        3072,
        kNN,
        't',
        'f',
        'f',
        dqf1_tf_nn,
        0,
        1), // mlp_proj_dgrad
    MIXED_ROW(
        4096, 768, 3072, kNT, 't', 'f', 'f', dnf1_tf_nt, 0, 1), // mlp_proj_fwd
    MIXED_ROW(3072,
        768,
        4096,
        kTN,
        't',
        'f',
        'f',
        dnc1_tf_tn,
        136,
        8), // mlp_proj_wgrad
};

#undef MIXED_ROW

//! The claim of a call, by exact shape, transposes and layout.
Claim const *find_claim(const Call &call) noexcept
{
    if (call.batch != 1)
    {
        return nullptr;
    }
    int const combo =
        (static_cast<int>(call.ta) << 1) | static_cast<int>(call.tb);
    int const layout = ac_of(call.a_shift) | (ac_of(call.b_shift) << 1) |
                       (ac_of(call.c_shift) << 2);
    for (Claim const &c : kClaims)
    {
        if (c.m == call.m && c.n == call.n && c.k == call.k &&
            c.combo == combo && c.layout == layout)
        {
            return &c;
        }
    }
    return nullptr;
}

//! C-order storage of A for a transpose combo: NoTrans is m x k
//! row-major, Trans is k x m column-major.
template <int Combo>
using LayoutAOf = md::cond_t<(Combo & 2) != 0, md::CM, md::RM>;
//! B: NoTrans is k x n row-major, Trans is n x k column-major.
template <int Combo>
using LayoutBOf = md::cond_t<(Combo & 1) != 0, md::CM, md::RM>;

//! Run one instantiation: the plain kernel over every output tile or,
//! under a split schedule, the plain kernel on the single-owner tiles,
//! then the piece kernel, then the reduction that finishes the split
//! tiles.
template <typename G, int Occup>
bool run_gemm(
    const Call &call, cudaStream_t stream, int full, int slices) noexcept
{
    using Mma = typename G::Mma;
    using Epilogue = typename G::Epilogue;
    using Plain = md::MixedKernel<Mma, Epilogue>;
    using Piece = md::MixedPieceKernel<Mma, Epilogue>;
    using OutIterator = typename Epilogue::OutputTileIterator;
    using LA = typename G::LayoutA;
    using LB = typename G::LayoutB;
    using RefA = typename Mma::IteratorA::TensorRef;
    using RefB = typename Mma::IteratorB::TensorRef;
    int const tiles_m = call.m / Plain::kM;
    int const tiles_n = call.n / Plain::kN;
    int const tiles = tiles_m * tiles_n;
    int const panels = call.k / Plain::kK;
    if (full < 0 || full > tiles || slices < 1)
    {
        return false;
    }
    bool const split = full < tiles && slices > 1;
    if (split && panels < slices)
    {
        return false;
    }
    // Partials of the split tiles: one contiguous kM x kN slot per piece,
    // in the shared grow-only piece scratch.
    float *partials = nullptr;
    if (split)
    {
        size_t const bytes = size_t(tiles - full) * size_t(slices) *
                             size_t(Plain::kM) * size_t(Plain::kN) *
                             sizeof(float);
        partials = static_cast<float *>(scratch(bytes));
        if (partials == nullptr)
        {
            return false;
        }
    }
    LA const la = LA::packed({call.m, call.k});
    LB const lb = LB::packed({call.k, call.n});
    md::RM const lc = md::RM::packed({call.m, call.n});
    int const plain_ctas = split ? full : tiles;
    if (plain_ctas > 0)
    {
        typename Plain::Params p;
        p.problem_size = cutlass::gemm::GemmCoord(call.m, call.n, call.k);
        p.tiles_m = tiles_m;
        p.tiles_n = tiles_n;
        p.ref_A =
            RefA(static_cast<typename RefA::Element *>(
                     const_cast<float *>(static_cast<float const *>(call.a))),
                la);
        p.params_A =
            md::iter_params<typename Mma::IteratorA>(la, call.a_shift);
        p.ref_B =
            RefB(static_cast<typename RefB::Element *>(
                     const_cast<float *>(static_cast<float const *>(call.b))),
                lb);
        p.params_B =
            md::iter_params<typename Mma::IteratorB>(lb, call.b_shift);
        p.ref_C =
            typename OutIterator::TensorRef(static_cast<float *>(call.c), lc);
        p.params_C = md::iter_params<OutIterator>(lc, call.c_shift);
        p.ref_D =
            typename OutIterator::TensorRef(static_cast<float *>(call.c), lc);
        p.params_D = md::iter_params<OutIterator>(lc, call.c_shift);
        p.output_op = typename Plain::OutputOp::Params(call.alpha, call.beta);
        if (!md::launch_kernel<Plain, Occup>(p, plain_ctas, stream))
        {
            return false;
        }
    }
    if (!split)
    {
        return true;
    }
    typename Piece::Params q;
    q.problem_size = cutlass::gemm::GemmCoord(call.m, call.n, call.k);
    q.tiles_m = tiles_m;
    q.split_full = full;
    q.split_slices = slices;
    q.ref_A = RefA(static_cast<typename RefA::Element *>(const_cast<float *>(
                       static_cast<float const *>(call.a))),
        la);
    q.params_A = md::iter_params<typename Mma::IteratorA>(la, call.a_shift);
    q.ref_B = RefB(static_cast<typename RefB::Element *>(const_cast<float *>(
                       static_cast<float const *>(call.b))),
        lb);
    q.params_B = md::iter_params<typename Mma::IteratorB>(lb, call.b_shift);
    q.params_ws = md::iter_params<OutIterator>(
        md::RM::packed({Plain::kM, Plain::kN}), 0);
    q.partials = partials;
    if (!md::launch_kernel<Piece, Occup>(q, (tiles - full) * slices, stream))
    {
        return false;
    }
    constexpr int kTile = Plain::kM * Plain::kN;
    constexpr int kThreads = 256;
    int const wide = call.n % 4 == 0 ? 4 : 1;
    long long const total =
        static_cast<long long>(tiles - full) * (kTile / wide);
    int const blocks =
        int(std::min<long long>((total + kThreads - 1) / kThreads, 4096));
    if (wide == 4)
    {
        md::mixed_reduce<Plain::kM, Plain::kN, true>
            <<<blocks, kThreads, 0, stream>>>(partials,
                call.c,
                call.c_shift,
                call.alpha,
                call.beta,
                call.n,
                tiles_m,
                full,
                tiles,
                slices);
    }
    else
    {
        md::mixed_reduce<Plain::kM, Plain::kN, false>
            <<<blocks, kThreads, 0, stream>>>(partials,
                call.c,
                call.c_shift,
                call.alpha,
                call.beta,
                call.n,
                tiles_m,
                full,
                tiles,
                slices);
    }
    return cudaGetLastError() == cudaSuccess;
}

//! One variant: the tile's shape law (the quad loaders do not predicate),
//! whole 8-panels in k, batch 1, a flat C for the native epilogue, and
//! the variant's addressing pair and transpose combo.
template <int Tile,
    int Native,
    int NativeC,
    int Occup,
    int ACA,
    int ACB,
    int Combo>
bool run_variant(
    const Call &call, cudaStream_t stream, int full, int slices) noexcept
{
    constexpr int kBM = Tile == 2 ? 64 : 128;
    constexpr int kBN = Tile == 1 ? 64 : 128;
    if (call.batch != 1 || call.m % kBM != 0 || call.n % kBN != 0 ||
        call.k % 8 != 0)
    {
        return false;
    }
    if (NativeC != 0 && call.c_shift != 0)
    {
        return false;
    }
    int const combo =
        (static_cast<int>(call.ta) << 1) | static_cast<int>(call.tb);
    if (ac_of(call.a_shift) != ACA || ac_of(call.b_shift) != ACB ||
        combo != Combo)
    {
        return false;
    }
    using Gemm = md::MixedGemm<LayoutAOf<Combo>,
        LayoutBOf<Combo>,
        kBM,
        kBN,
        32,
        64,
        ACA,
        ACB,
        Native != 0,
        NativeC != 0>;
    return run_gemm<Gemm, Occup>(call, stream, full, slices);
}

} // namespace

int mixed_pick(int base, const Call &call) noexcept
{
    Claim const *c = find_claim(call);
    return c == nullptr ? -1 : base + c->cfg;
}

const char *mixed_name(int cfg) noexcept
{
#define MIXED_NAMESTR(nm, tile, nat, natc, occ, aca, acb, combo) #nm,
    static char const *const kNames[] = {
        MIXED_VARIANTS(MIXED_NAMESTR) nullptr};
#undef MIXED_NAMESTR
    static char bufs[4][64];
    static int idx = 0;
    if (cfg < 0 || cfg >= kNumVariants)
    {
        return "chunked-ext-mixed-?";
    }
    std::snprintf(
        bufs[idx], sizeof(bufs[idx]), "chunked-ext-mixed-%s", kNames[cfg]);
    char *buf = bufs[idx];
    idx = (idx + 1) & 3;
    return buf;
}

bool mixed_launch(int cfg, const Call &call, cudaStream_t stream) noexcept
{
    if (cfg < 0 || cfg >= kNumVariants)
    {
        return false;
    }
    // The split schedule of an exact claim; any other call (a pinned
    // probe) runs the kernel unsplit.
    int full = 0;
    int slices = 1;
    if (Claim const *c = find_claim(call); c != nullptr && c->cfg == cfg)
    {
        full = c->full;
        slices = c->slices;
    }
    switch (cfg)
    {
#define MIXED_CASE(nm, tile, nat, natc, occ, aca, acb, combo)                 \
    case nm:                                                                  \
        return run_variant<tile, nat, natc, occ, aca, acb, combo>(            \
            call, stream, full, slices);
        MIXED_VARIANTS(MIXED_CASE)
#undef MIXED_CASE
    default:
        return false;
    }
}

} // namespace ext

} // namespace nntile::kernel::gemm_chunked

#endif // NNTILE_USE_CUDA
