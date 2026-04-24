/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/model/llama/llama_attention.cc
 * Tests for LlamaAttention (sdpa_eager-based).
 *
 * Tensor shapes (hidden, seq, batch) and head geometry match
 * ``wrappers/python/tests/model/test_llama_attention.py`` (``single_tile``):
 * head size 64, ``seq_len`` 64, ``n_batch`` 3; MHA (1,1) uses ``hidden=64``;
 * GQA (8,4) uses ``hidden=512``. Safetensors from
 * ``tests/graph/model/llama/generate_test_data.py`` (``ATTENTION_*_DIMS``).
 * RoPE/causal-mask reference bundles: default ``llama_attention(_gqa)_full`` plus
 * six extras from ``--write-attention-rope-mask-variants``. Catch tags:
 * ``[nomask]`` — no causal ``attn_mask`` (RoPE and no-RoPE bundles);
 * ``[causal_mask]`` — causal ``attn_mask``;
 * ``[norope]`` — no-RoPE bundles only (with or without causal mask);
 * ``[norope_nomask]`` — no-RoPE and no causal mask (subset of ``[nomask]``).
 * Run e.g. ``./test_llama_attention '[nomask]'`` or ``'[norope_nomask]'``.
 *
 * NNTile tensor **storage** is Fortran (column-major) everywhere, including
 * ``bind_hint`` bytes from safetensors. ``rotate_tensor_in`` /
 * ``rotate_tensor_out`` in ``wrappers/python/nntile/model/llama_attention.py``
 * are implemented on **NumPy default (C / row-major) views**: ``reshape`` and
 * ``moveaxis`` use C element order. To match that reference exactly, the test
 * helpers decode Fortran bytes to a temporary logical buffer in **C flat
 * layout**, apply the same reshape / moveaxis / rotate there, then encode back
 * to Fortran for ``set_bind_hint``. The graph still sees only Fortran-order
 * weights; the C buffer is not a different NNTile convention, only a bridge for
 * NumPy-identical index arithmetic.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/llama/llama_attention.hh"
#include "nntile/graph/model/llama/llama_config.hh"
#include "nntile/graph/tensor/fill.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::model::llama;
using namespace nntile::graph::io;
namespace gt = nntile::graph::tensor;

// Matches test_llama_attention.LlamaAttentionTestParams (single_tile).
static constexpr Index kAttnHeadSize = 64;
static constexpr Index kAttnSeq = 64;
static constexpr Index kAttnBatch = 3;

// MHA: 1 Q head, 1 KV head -> hidden = head_size.
static constexpr Index kMhaHidden = kAttnHeadSize;

// GQA: 8 Q heads, 4 KV heads -> hidden = 8 * head_size.
static constexpr Index kGqaNumHeads = 8;
static constexpr Index kGqaNumKv = 4;
static constexpr Index kGqaHidden = kAttnHeadSize * kGqaNumHeads;

static LlamaConfig test_config()
{
    LlamaConfig config;
    config.hidden_size = kMhaHidden;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.compute_head_dim();
    return config;
}

static LlamaConfig test_config_gqa()
{
    LlamaConfig config;
    config.hidden_size = kGqaHidden;
    config.num_attention_heads = kGqaNumHeads;
    config.num_key_value_heads = kGqaNumKv;
    config.compute_head_dim();
    return config;
}

namespace
{

//! Q/K weight rotation for tests. NNTile data remain Fortran on the wire; see
//! file comment: we mirror NumPy C-order ``reshape`` / ``moveaxis`` /
//! ``rotate_tensor_in`` / ``rotate_tensor_out`` from ``llama_attention.py``.
namespace rotate_qk_weight
{

static Index product_shape(
    const std::vector<Index>& shape, int lo, int hi)
{
    Index p = 1;
    for(int a = lo; a < hi; ++a)
    {
        p *= shape[static_cast<size_t>(a)];
    }
    return p;
}

static Index nelems_shape(const std::vector<Index>& shape)
{
    return product_shape(shape, 0, static_cast<int>(shape.size()));
}

//! Column-major (Fortran) flat index -> linear fi in [0, nelems).
static void fortran_unravel(
    Index fi,
    const std::vector<Index>& shape,
    std::vector<Index>& idx)
{
    Index rem = fi;
    for(size_t k = 0; k < shape.size(); ++k)
    {
        idx[k] = rem % shape[k];
        rem /= shape[k];
    }
}

//! Row-major (C) flat index from multi-index ``idx``.
static Index index_c_rowmajor(
    const std::vector<Index>& shape,
    const std::vector<Index>& idx)
{
    Index flat = 0;
    Index stride = 1;
    for(size_t k = shape.size(); k-- > 0;)
    {
        flat += idx[k] * stride;
        stride *= shape[k];
    }
    return flat;
}

static std::vector<float> fortran_to_c_float(
    const std::vector<Index>& shape,
    const float* f_src)
{
    const Index n = nelems_shape(shape);
    std::vector<float> c(static_cast<size_t>(n));
    std::vector<Index> idx(shape.size());
    for(Index fi = 0; fi < n; ++fi)
    {
        fortran_unravel(fi, shape, idx);
        const Index ci = index_c_rowmajor(shape, idx);
        c[static_cast<size_t>(ci)] = f_src[static_cast<size_t>(fi)];
    }
    return c;
}

static void c_to_fortran_float(
    const std::vector<Index>& shape,
    const float* c_src,
    float* f_dst)
{
    const Index n = nelems_shape(shape);
    std::vector<Index> idx(shape.size());
    for(Index fi = 0; fi < n; ++fi)
    {
        fortran_unravel(fi, shape, idx);
        const Index ci = index_c_rowmajor(shape, idx);
        f_dst[static_cast<size_t>(fi)] = c_src[static_cast<size_t>(ci)];
    }
}

//! ``numpy.moveaxis(arr, 0, 1)`` with C-order data; ``sh_in`` is the *input*
//! shape (first two dims will be swapped in the output layout).
static std::vector<float> moveaxis_swap01_c(
    const std::vector<float>& src,
    const std::vector<Index>& sh_in)
{
    if(sh_in.size() < 2)
    {
        return src;
    }
    std::vector<Index> sh_out = sh_in;
    std::swap(sh_out[0], sh_out[1]);
    const Index n = nelems_shape(sh_in);
    std::vector<float> dst(static_cast<size_t>(n));
    std::vector<Index> out_idx(sh_out.size());
    std::vector<Index> in_idx(sh_in.size());
    const int R = static_cast<int>(sh_out.size());
    for(Index flat_out = 0; flat_out < n; ++flat_out)
    {
        Index rem = flat_out;
        for(int k = R - 1; k >= 0; --k)
        {
            out_idx[static_cast<size_t>(k)] =
                rem % sh_out[static_cast<size_t>(k)];
            rem /= sh_out[static_cast<size_t>(k)];
        }
        in_idx[0] = out_idx[1];
        in_idx[1] = out_idx[0];
        for(int k = 2; k < R; ++k)
        {
            in_idx[static_cast<size_t>(k)] = out_idx[static_cast<size_t>(k)];
        }
        dst[static_cast<size_t>(index_c_rowmajor(sh_out, out_idx))] =
            src[static_cast<size_t>(index_c_rowmajor(sh_in, in_idx))];
    }
    return dst;
}

//! C-order 3D collapse of ``shape`` at ``axis``; reads only ``src``, writes a
//! fresh buffer (same idea as ``np.empty_like`` + slice assign in
//! ``llama_attention.rotate_tensor_in``).
static std::vector<float> rotate_tensor_in(
    const std::vector<float>& src,
    const std::vector<Index>& shape,
    int axis)
{
    const int nd = static_cast<int>(shape.size());
    if(axis < 0 || axis >= nd)
    {
        throw std::runtime_error("rotate_tensor_in: bad axis");
    }
    const Index n0 = product_shape(shape, 0, axis);
    const Index n1 = shape[static_cast<size_t>(axis)];
    const Index n2 = product_shape(shape, axis + 1, nd);
    const Index n = n0 * n1 * n2;
    if(static_cast<Index>(src.size()) != n)
    {
        throw std::runtime_error("rotate_tensor_in: size mismatch");
    }
    auto flat3 = [n0, n1, n2](Index i0, Index i1, Index i2) -> Index {
        return i2 + n2 * (i1 + n1 * i0);
    };
    std::vector<float> dst(static_cast<size_t>(n));
    const Index mid = n1 / 2;
    for(Index i0 = 0; i0 < n0; ++i0)
    {
        for(Index j = 0; j < mid; ++j)
        {
            for(Index i2 = 0; i2 < n2; ++i2)
            {
                dst[static_cast<size_t>(flat3(i0, 2 * j, i2))] =
                    src[static_cast<size_t>(flat3(i0, j, i2))];
                dst[static_cast<size_t>(flat3(i0, 2 * j + 1, i2))] =
                    src[static_cast<size_t>(flat3(i0, mid + j, i2))];
            }
        }
    }
    return dst;
}

static std::vector<float> rotate_tensor_out(
    const std::vector<float>& src,
    const std::vector<Index>& shape,
    int axis)
{
    const int nd = static_cast<int>(shape.size());
    if(axis < 0 || axis >= nd)
    {
        throw std::runtime_error("rotate_tensor_out: bad axis");
    }
    const Index n0 = product_shape(shape, 0, axis);
    const Index n1 = shape[static_cast<size_t>(axis)];
    const Index n2 = product_shape(shape, axis + 1, nd);
    const Index n = n0 * n1 * n2;
    if(static_cast<Index>(src.size()) != n)
    {
        throw std::runtime_error("rotate_tensor_out: size mismatch");
    }
    auto flat3 = [n0, n1, n2](Index i0, Index i1, Index i2) -> Index {
        return i2 + n2 * (i1 + n1 * i0);
    };
    std::vector<float> dst(static_cast<size_t>(n));
    const Index mid = n1 / 2;
    for(Index i0 = 0; i0 < n0; ++i0)
    {
        for(Index j = 0; j < mid; ++j)
        {
            for(Index i2 = 0; i2 < n2; ++i2)
            {
                dst[static_cast<size_t>(flat3(i0, j, i2))] =
                    src[static_cast<size_t>(flat3(i0, 2 * j, i2))];
                dst[static_cast<size_t>(flat3(i0, mid + j, i2))] =
                    src[static_cast<size_t>(flat3(i0, 2 * j + 1, i2))];
            }
        }
    }
    return dst;
}

static void apply_q_weight_rotate_in(NNGraph::TensorNode* w)
{
    const auto& sh = w->shape();
    if(sh.size() != 3 && sh.size() != 4)
    {
        return;
    }
    const auto* hint = w->data()->get_bind_hint();
    if(hint == nullptr || hint->size() % sizeof(float) != 0)
    {
        return;
    }
    const size_t nb = hint->size();
    std::vector<float> fbuf(nb / sizeof(float));
    std::memcpy(
        fbuf.data(), hint->data(), nb);
    std::vector<float> c = fortran_to_c_float(sh, fbuf.data());
    std::vector<Index> tmp_sh = sh;
    std::swap(tmp_sh[0], tmp_sh[1]);
    std::vector<float> mid = moveaxis_swap01_c(c, tmp_sh);
    const std::vector<float> rotated = rotate_tensor_in(mid, sh, 2);
    std::vector<float> f_new(nb / sizeof(float));
    c_to_fortran_float(sh, rotated.data(), f_new.data());
    std::vector<std::uint8_t> bytes(nb);
    std::memcpy(bytes.data(), f_new.data(), nb);
    w->data()->set_bind_hint(std::move(bytes));
}

static void apply_q_weight_rotate_out(NNGraph::TensorNode* w)
{
    const auto& sh = w->shape();
    if(sh.size() != 3 && sh.size() != 4)
    {
        return;
    }
    const auto* hint = w->data()->get_bind_hint();
    if(hint == nullptr || hint->size() % sizeof(float) != 0)
    {
        return;
    }
    const size_t nb = hint->size();
    std::vector<float> fbuf(nb / sizeof(float));
    std::memcpy(fbuf.data(), hint->data(), nb);
    std::vector<float> c = fortran_to_c_float(sh, fbuf.data());
    const std::vector<float> unrot = rotate_tensor_out(c, sh, 2);
    std::vector<Index> tmp_sh = sh;
    std::swap(tmp_sh[0], tmp_sh[1]);
    std::vector<float> restored = moveaxis_swap01_c(unrot, sh);
    std::vector<float> f_new(nb / sizeof(float));
    c_to_fortran_float(sh, restored.data(), f_new.data());
    std::vector<std::uint8_t> bytes(nb);
    std::memcpy(bytes.data(), f_new.data(), nb);
    w->data()->set_bind_hint(std::move(bytes));
}

static void apply_k_weight_rotate_in(NNGraph::TensorNode* w)
{
    const auto& sh = w->shape();
    if(sh.size() != 3)
    {
        return;
    }
    const auto* hint = w->data()->get_bind_hint();
    if(hint == nullptr || hint->size() % sizeof(float) != 0)
    {
        return;
    }
    const size_t nb = hint->size();
    std::vector<float> fbuf(nb / sizeof(float));
    std::memcpy(fbuf.data(), hint->data(), nb);
    std::vector<float> c = fortran_to_c_float(sh, fbuf.data());
    const std::vector<float> rotated = rotate_tensor_in(c, sh, 1);
    std::vector<float> f_new(nb / sizeof(float));
    c_to_fortran_float(sh, rotated.data(), f_new.data());
    std::vector<std::uint8_t> bytes(nb);
    std::memcpy(bytes.data(), f_new.data(), nb);
    w->data()->set_bind_hint(std::move(bytes));
}

static void apply_k_weight_rotate_out(NNGraph::TensorNode* w)
{
    const auto& sh = w->shape();
    if(sh.size() != 3)
    {
        return;
    }
    const auto* hint = w->data()->get_bind_hint();
    if(hint == nullptr || hint->size() % sizeof(float) != 0)
    {
        return;
    }
    const size_t nb = hint->size();
    std::vector<float> fbuf(nb / sizeof(float));
    std::memcpy(fbuf.data(), hint->data(), nb);
    std::vector<float> c = fortran_to_c_float(sh, fbuf.data());
    const std::vector<float> unrot = rotate_tensor_out(c, sh, 1);
    std::vector<float> f_new(nb / sizeof(float));
    c_to_fortran_float(sh, unrot.data(), f_new.data());
    std::vector<std::uint8_t> bytes(nb);
    std::memcpy(bytes.data(), f_new.data(), nb);
    w->data()->set_bind_hint(std::move(bytes));
}

} // namespace rotate_qk_weight

//! Apply ``rotate_tensor_in`` to Q/K weights (Python ``from_torch`` path).
static void llama_attention_tests_rotate_qk_weights_in(
    graph::module::Module& mod)
{
    for(const auto& [name, t] : mod.named_parameters_recursive())
    {
        if(name.size() < 9)
        {
            continue;
        }
        if(name.compare(name.size() - 9, 9, ".q_weight") == 0)
        {
            rotate_qk_weight::apply_q_weight_rotate_in(t);
        }
        else if(name.compare(name.size() - 9, 9, ".k_weight") == 0)
        {
            rotate_qk_weight::apply_k_weight_rotate_in(t);
        }
    }
}

//! Reverse ``rotate_tensor_in`` on Q/K (Python ``to_torch`` path) for I/O.
static void llama_attention_tests_rotate_qk_weights_out(
    graph::module::Module& mod)
{
    for(const auto& [name, t] : mod.named_parameters_recursive())
    {
        if(name.size() < 9)
        {
            continue;
        }
        if(name.compare(name.size() - 9, 9, ".q_weight") == 0)
        {
            rotate_qk_weight::apply_q_weight_rotate_out(t);
        }
        else if(name.compare(name.size() - 9, 9, ".k_weight") == 0)
        {
            rotate_qk_weight::apply_k_weight_rotate_out(t);
        }
    }
}

//! Optional RoPE tensors from safetensors (same layout as Python LlamaAttention).
struct LlamaRopeInputs
{
    NNGraph::TensorNode* sin = nullptr;
    NNGraph::TensorNode* cos = nullptr;
    std::vector<float> sin_data;
    std::vector<float> cos_data;
};

inline bool load_llama_rope_inputs(
    NNGraph& g,
    const SafeTensorsReader& reader,
    const LlamaConfig& config,
    Index n_seq,
    Index n_batch,
    LlamaRopeInputs& out)
{
    out = {};
    if(!reader.has_tensor("rope_sin") || !reader.has_tensor("rope_cos"))
    {
        return false;
    }
    const Index head_dim = config.head_dim;
    if(head_dim % 2 != 0)
    {
        return false;
    }
    const Index half = head_dim / 2;
    out.sin = g.tensor({half, n_seq, n_batch}, "rope_sin", DataType::FP32);
    out.cos = g.tensor({half, n_seq, n_batch}, "rope_cos", DataType::FP32);
    auto read_f = [&](const char* name, std::vector<float>& dst)
    {
        std::vector<std::uint8_t> b = reader.read_tensor(name);
        dst.resize(b.size() / sizeof(float));
        std::memcpy(dst.data(), b.data(), b.size());
    };
    read_f("rope_sin", out.sin_data);
    read_f("rope_cos", out.cos_data);
    return true;
}

inline void mark_rope_inputs(const LlamaRopeInputs& rope)
{
    if(rope.sin == nullptr)
    {
        return;
    }
    rope.sin->mark_input(true);
    rope.cos->mark_input(true);
}

inline void bind_rope_inputs(
    TileGraph::Runtime& runtime, const LlamaRopeInputs& rope)
{
    if(rope.sin == nullptr)
    {
        return;
    }
    runtime.bind_data("rope_sin", rope.sin_data);
    runtime.bind_data("rope_cos", rope.cos_data);
}

//! Optional causal mask ``(seq, seq)`` for ``sdpa_eager`` (1 = keep logit).
//! Safetensors store float32 0/1 (``save_file`` maps numpy bool to F32); BOOL
//! is also accepted. Bytes bound to the graph are BOOL layout (1 byte/elem).
inline bool load_attn_mask_bool(
    NNGraph& g,
    const SafeTensorsReader& reader,
    Index n_seq,
    NNGraph::TensorNode*& out_mask,
    std::vector<std::uint8_t>& mask_bytes)
{
    out_mask = nullptr;
    mask_bytes.clear();
    if(!reader.has_tensor("attn_mask"))
    {
        return false;
    }
    const auto& info = reader.tensor_info("attn_mask");
    if(info.shape.size() != 2 || info.shape[0] != n_seq
        || info.shape[1] != n_seq)
    {
        throw std::runtime_error("Llama attention test: attn_mask shape mismatch");
    }
    const auto n_el = static_cast<size_t>(n_seq * n_seq);
    out_mask = g.tensor({n_seq, n_seq}, "attn_mask", DataType::BOOL, false);
    auto raw = reader.read_tensor("attn_mask");
    if(info.dtype == DataType::BOOL)
    {
        if(raw.size() != n_el)
        {
            throw std::runtime_error(
                "Llama attention test: BOOL attn_mask byte size mismatch");
        }
        mask_bytes = std::move(raw);
        return true;
    }
    if(info.dtype == DataType::FP32)
    {
        if(raw.size() != n_el * sizeof(float))
        {
            throw std::runtime_error(
                "Llama attention test: F32 attn_mask byte size mismatch");
        }
        mask_bytes.resize(n_el);
        const auto* p = reinterpret_cast<const float*>(raw.data());
        for(size_t i = 0; i < n_el; ++i)
        {
            mask_bytes[i] =
                (p[i] > 0.5f) ? static_cast<std::uint8_t>(1)
                              : static_cast<std::uint8_t>(0);
        }
        return true;
    }
    throw std::runtime_error(
        "Llama attention test: attn_mask must be BOOL or F32");
}

inline void mark_mask_input(NNGraph::TensorNode* mask)
{
    if(mask != nullptr)
    {
        mask->mark_input(true);
    }
}

inline void bind_mask_input(
    TileGraph::Runtime& runtime,
    NNGraph::TensorNode* mask,
    const std::vector<std::uint8_t>& mask_bytes)
{
    if(mask == nullptr)
    {
        return;
    }
    runtime.bind_data(mask->name(), mask_bytes);
}

} // namespace

TEST_CASE("LlamaAttention forward builds output", "[model][llama]")
{
    NNGraph g("llama_attn");
    auto config = test_config();

    auto* input = g.tensor(
        {kMhaHidden, kAttnSeq, kAttnBatch}, "input", DataType::FP32);
    LlamaAttention attn(&g, "attn", config);
    auto* output = attn.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({
        kMhaHidden, kAttnSeq, kAttnBatch}));
}

TEST_CASE("LlamaAttention GQA forward builds output", "[model][llama][gqa]")
{
    NNGraph g("llama_attn_gqa");
    auto config = test_config_gqa();

    auto* input = g.tensor(
        {kGqaHidden, kAttnSeq, kAttnBatch}, "input", DataType::FP32);
    LlamaAttention attn(&g, "attn", config);
    auto* output = attn.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() == std::vector<Index>({
        kGqaHidden, kAttnSeq, kAttnBatch}));
}

#ifdef LLAMA_DATA_DIR
TEST_CASE("LlamaAttention load from safetensors roundtrip", "[model][llama][io]")
{
    const std::string data_path =
        std::string(LLAMA_DATA_DIR) + "/llama_attention_full.safetensors";
    std::ifstream check(data_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    auto config = test_config();

    NNGraph g1("load_graph");
    LlamaAttention attn1(&g1, "attn", config);
    attn1.load(data_path);
    SafeTensorsReader rround(data_path);
    if(rround.has_tensor("rope_sin") && rround.has_tensor("rope_cos"))
    {
        llama_attention_tests_rotate_qk_weights_in(attn1);
        llama_attention_tests_rotate_qk_weights_out(attn1);
    }

    const std::string save_path =
        "/tmp/nntile_llama_attn_roundtrip.safetensors";
    attn1.save(save_path);

    SafeTensorsReader reader(data_path);
    SafeTensorsReader reader2(save_path);

    for(const auto& name : reader2.tensor_names())
    {
        REQUIRE(reader.has_tensor(name));
        auto orig = reader.read_tensor(name);
        auto loaded = reader2.read_tensor(name);
        REQUIRE(orig.size() == loaded.size());
        REQUIRE(orig == loaded);
    }

    std::remove(save_path.c_str());
}

namespace
{

//! MHA forward vs ``output_ref`` in ``full_path`` (must exist); ``fname`` names the graph.
void llama_mha_forward_compare_ref(
    const std::string& full_path,
    const char* fname,
    float tol)
{
    auto config = test_config();

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> result;
    {
        const std::string gname = std::string("attn_ref_") + fname;
        NNGraph g(gname);
        auto* input = g.tensor(
            {kMhaHidden, kAttnSeq, kAttnBatch}, "input", DataType::FP32);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(
            g, reader, config, kAttnSeq, kAttnBatch, rope);
        NNGraph::TensorNode* mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, kAttnSeq, mask, mask_bytes);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, mask);
        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        attn.load(full_path);
        if(reader.has_tensor("rope_sin") && reader.has_tensor("rope_cos"))
        {
            llama_attention_tests_rotate_qk_weights_in(attn);
        }

        TensorGraph& tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("input", input_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output->name());
    }

    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, tol);
}

void llama_mha_backward_vs_ref(
    const std::string& full_path,
    const char* fname,
    float tol)
{
    auto config = test_config();

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> grad_out_bytes = reader.read_tensor("grad_output");
    std::vector<float> grad_out_data(grad_out_bytes.size() / sizeof(float));
    std::memcpy(grad_out_data.data(), grad_out_bytes.data(),
        grad_out_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("grad_input");
    std::vector<float> grad_input_ref(ref_bytes.size() / sizeof(float));
    std::memcpy(grad_input_ref.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> grad_input_result;
    {
        const std::string gname = std::string("attn_bwd_") + fname;
        NNGraph g(gname);
        auto* input = g.tensor(
            {kMhaHidden, kAttnSeq, kAttnBatch}, "input", DataType::FP32, true);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(
            g, reader, config, kAttnSeq, kAttnBatch, rope);
        NNGraph::TensorNode* mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, kAttnSeq, mask, mask_bytes);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, mask);

        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        attn.load(full_path);
        if(reader.has_tensor("rope_sin") && reader.has_tensor("rope_cos"))
        {
            llama_attention_tests_rotate_qk_weights_in(attn);
        }

        TensorGraph& tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.bind_data("grad_output", grad_out_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        grad_input_result =
            runtime.get_output<float>(input->grad()->name());
    }

    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    require_relative_frobenius_error(grad_input_result, grad_input_ref, tol);
}

void llama_gqa_forward_compare_ref(
    const std::string& full_path,
    const char* fname,
    float tol)
{
    auto config = test_config_gqa();

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> result;
    {
        const std::string gname = std::string("attn_gqa_ref_") + fname;
        NNGraph g(gname);
        auto* input = g.tensor(
            {kGqaHidden, kAttnSeq, kAttnBatch}, "input", DataType::FP32);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(
            g, reader, config, kAttnSeq, kAttnBatch, rope);
        NNGraph::TensorNode* mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, kAttnSeq, mask, mask_bytes);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, mask);
        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        attn.load(full_path);
        if(reader.has_tensor("rope_sin") && reader.has_tensor("rope_cos"))
        {
            llama_attention_tests_rotate_qk_weights_in(attn);
        }

        TensorGraph& tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("input", input_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output->name());
    }

    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, tol);
}

void llama_gqa_backward_compare_ref(
    const std::string& full_path,
    const char* fname,
    float tol)
{
    auto config = test_config_gqa();

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> grad_out_bytes = reader.read_tensor("grad_output");
    std::vector<float> grad_out_data(grad_out_bytes.size() / sizeof(float));
    std::memcpy(grad_out_data.data(), grad_out_bytes.data(),
        grad_out_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("grad_input");
    std::vector<float> grad_input_ref(ref_bytes.size() / sizeof(float));
    std::memcpy(grad_input_ref.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> grad_input_result;
    {
        const std::string gname = std::string("attn_gqa_bwd_") + fname;
        NNGraph g(gname);
        auto* input = g.tensor(
            {kGqaHidden, kAttnSeq, kAttnBatch}, "input", DataType::FP32, true);
        LlamaRopeInputs rope;
        load_llama_rope_inputs(
            g, reader, config, kAttnSeq, kAttnBatch, rope);
        NNGraph::TensorNode* mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        load_attn_mask_bool(g, reader, kAttnSeq, mask, mask_bytes);
        LlamaAttention attn(&g, "attn", config);
        auto* output = attn.forward(input, rope.sin, rope.cos, mask);

        input->mark_input(true);
        output->mark_output(true);
        mark_rope_inputs(rope);
        mark_mask_input(mask);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        attn.load(full_path);
        if(reader.has_tensor("rope_sin") && reader.has_tensor("rope_cos"))
        {
            llama_attention_tests_rotate_qk_weights_in(attn);
        }

        TensorGraph& tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

        TileGraph::Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data("input", input_data);
        runtime.bind_data("grad_output", grad_out_data);
        bind_rope_inputs(runtime, rope);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        grad_input_result =
            runtime.get_output<float>(input->grad()->name());
    }

    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    require_relative_frobenius_error(grad_input_result, grad_input_ref, tol);
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA forward vs PyTorch (no causal mask, RoPE)",
    "[model][llama][nomask]")
{
    const char* fname = "llama_attention_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_mha_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA forward vs PyTorch (no causal mask, no RoPE)",
    "[model][llama][nomask][norope][norope_nomask]")
{
    const char* fname = "llama_attention_no_rope_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_mha_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA forward vs PyTorch (causal mask, RoPE)",
    "[model][llama][causal_mask]")
{
    const char* fname = "llama_attention_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 5e-3f;
    llama_mha_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA forward vs PyTorch (causal mask, no RoPE)",
    "[model][llama][causal_mask][norope]")
{
    const char* fname = "llama_attention_no_rope_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_mha_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA backward vs PyTorch (no causal mask, RoPE)",
    "[model][llama][nomask]")
{
    const char* fname = "llama_attention_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 9e-3f;
    llama_mha_backward_vs_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA backward vs PyTorch (no causal mask, no RoPE)",
    "[model][llama][nomask][norope][norope_nomask]")
{
    const char* fname = "llama_attention_no_rope_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_mha_backward_vs_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA backward vs PyTorch (causal mask, RoPE)",
    "[model][llama][causal_mask]")
{
    const char* fname = "llama_attention_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 9e-3f;
    llama_mha_backward_vs_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention MHA backward vs PyTorch (causal mask, no RoPE)",
    "[model][llama][causal_mask][norope]")
{
    const char* fname = "llama_attention_no_rope_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_mha_backward_vs_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA forward vs PyTorch (no causal mask, RoPE)",
    "[model][llama][gqa][nomask]")
{
    const char* fname = "llama_attention_gqa_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA test data not found.");
    }

    constexpr float tol = 5e-3f;
    llama_gqa_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA forward vs PyTorch (no causal mask, no RoPE)",
    "[model][llama][gqa][nomask][norope][norope_nomask]")
{
    const char* fname = "llama_attention_gqa_no_rope_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_gqa_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA forward vs PyTorch (causal mask, RoPE)",
    "[model][llama][gqa][causal_mask]")
{
    const char* fname = "llama_attention_gqa_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA test data not found.");
    }

    constexpr float tol = 5e-3f;
    llama_gqa_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA forward vs PyTorch (causal mask, no RoPE)",
    "[model][llama][gqa][causal_mask][norope]")
{
    const char* fname = "llama_attention_gqa_no_rope_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_gqa_forward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward vs PyTorch (no causal mask, RoPE)",
    "[model][llama][gqa][nomask]")
{
    const char* fname = "llama_attention_gqa_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA backward test data not found.");
    }

    constexpr float tol = 9e-3f;
    llama_gqa_backward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward vs PyTorch (no causal mask, no RoPE)",
    "[model][llama][gqa][nomask][norope][norope_nomask]")
{
    const char* fname = "llama_attention_gqa_no_rope_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA backward test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_gqa_backward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward vs PyTorch (causal mask, RoPE)",
    "[model][llama][gqa][causal_mask]")
{
    const char* fname = "llama_attention_gqa_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA backward test data not found.");
    }

    constexpr float tol = 9e-3f;
    llama_gqa_backward_compare_ref(full_path, fname, tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "LlamaAttention GQA backward vs PyTorch (causal mask, no RoPE)",
    "[model][llama][gqa][causal_mask][norope]")
{
    const char* fname = "llama_attention_gqa_no_rope_causal_full.safetensors";
    const std::string full_path =
        std::string(LLAMA_DATA_DIR) + "/" + fname;
    std::ifstream check(full_path);
    if(!check.good())
    {
        SKIP("Llama attention GQA backward test data not found.");
    }

    constexpr float tol = 1e-6f;
    llama_gqa_backward_compare_ref(full_path, fname, tol);
}
#endif
