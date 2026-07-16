/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/tests/models_fwd_bwd_smoke.cc
 * C++ forward/backward smoke for every model and major submodule.
 *
 * No numerical accuracy checks — only that shapes / devices are consistent
 * and autograd completes without raising.
 */

#include "model_smoke_helpers.hh"

#include <torch_nntile/torch_nntile.hh>

#include <catch2/catch_test_macros.hpp>

namespace
{

using namespace torch_nntile;
using namespace torch_nntile::models;
using namespace torch_nntile::test;

constexpr int64_t kB = 2;
constexpr int64_t kS = 8;
constexpr int64_t kV = 128;
constexpr int64_t kH = 64;
constexpr int64_t kI = 128;
constexpr int64_t kHeads = 4;
constexpr int64_t kHd = kH / kHeads; // 16

LlamaConfig tiny_llama()
{
    LlamaConfig cfg;
    cfg.vocab_size = kV;
    cfg.hidden_size = kH;
    cfg.intermediate_size = kI;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = kHeads;
    cfg.num_key_value_heads = kHeads;
    cfg.max_position_embeddings = 32;
    return cfg;
}

BertConfig tiny_bert()
{
    BertConfig cfg;
    cfg.vocab_size = kV;
    cfg.hidden_size = kH;
    cfg.intermediate_size = kI;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = kHeads;
    cfg.max_position_embeddings = 32;
    return cfg;
}

RobertaConfig tiny_roberta()
{
    RobertaConfig cfg;
    cfg.vocab_size = kV;
    cfg.hidden_size = kH;
    cfg.intermediate_size = kI;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = kHeads;
    cfg.max_position_embeddings = 32;
    cfg.pad_token_id = 1;
    return cfg;
}

GptNeoConfig tiny_gpt_neo()
{
    GptNeoConfig cfg;
    cfg.vocab_size = kV;
    cfg.hidden_size = kH;
    cfg.intermediate_size = kI;
    cfg.num_hidden_layers = 2;
    cfg.num_attention_heads = kHeads;
    cfg.max_position_embeddings = 32;
    cfg.window_size = 4;
    return cfg;
}

GptNeoXConfig tiny_gpt_neox()
{
    GptNeoXConfig cfg;
    cfg.vocab_size = kV;
    cfg.hidden_size = kH;
    cfg.intermediate_size = kI;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = kHeads;
    cfg.max_position_embeddings = 32;
    // Full-head RoPE avoids aten::narrow (no nntile autograd yet).
    cfg.rotary_pct = 1.0;
    return cfg;
}

Gpt2Config tiny_gpt2()
{
    Gpt2Config cfg;
    cfg.vocab_size = kV;
    cfg.n_embd = kH;
    cfg.n_head = kHeads;
    cfg.n_layer = 1;
    cfg.n_positions = 32;
    cfg.n_inner = kI;
    return cfg;
}

T5Config tiny_t5()
{
    T5Config cfg;
    cfg.vocab_size = kV;
    cfg.d_model = kH;
    cfg.d_kv = kHd;
    cfg.d_ff = kI;
    cfg.num_layers = 1;
    cfg.num_decoder_layers = 1;
    cfg.num_heads = kHeads;
    return cfg;
}

at::Tensor rand_ids()
{
    return torch::randint(
        /*low=*/0,
        /*high=*/kV,
        {kB, kS},
        torch::TensorOptions().dtype(torch::kLong));
}

at::Tensor rand_hidden(bool requires_grad = true)
{
    return torch::randn(
        {kB, kS, kH},
        torch::TensorOptions().dtype(torch::kFloat32))
        .set_requires_grad(requires_grad);
}

} // namespace

// ---------------------------------------------------------------------------
// Full models
// ---------------------------------------------------------------------------

TEST_CASE("C++ DeepReLU forward smoke", "[models][smoke]")
{
    // DeepReLU uses torch::nn::Linear (aten matmul). Forward works on
    // nntile; matmul_backward is not implemented yet — smoke forward only.
    ContextGuard guard;
    auto model = DeepReLUImpl::tiny();
    auto x = to_nntile_float(
        torch::randn({4, 128}),
        /*requires_grad=*/false);
    assert_module_fwd_bwd_smoke(
        *model,
        [&]() { return model->forward(x); },
        {},
        {4, 10},
        /*require_backward=*/false);
}

TEST_CASE("C++ MlpMixer fwd+bwd smoke", "[models][smoke]")
{
    ContextGuard guard;
    MlpMixerConfig cfg;
    cfg.channel_dim = 8;
    cfg.init_patch_dim = 4;
    cfg.projected_patch_dim = 4;
    cfg.num_mixer_layers = 2;
    cfg.n_classes = 3;
    auto model = MlpMixer(cfg);
    auto x = to_nntile_float(
        torch::randn({8, kB, 4}),
        /*requires_grad=*/true);
    assert_module_fwd_bwd_smoke(
        *model,
        [&]() { return model->forward(x); },
        {x},
        {kB, 3});
}

TEST_CASE("C++ LlamaCausal fwd+bwd smoke", "[models][smoke]")
{
    ContextGuard guard;
    auto model = LlamaCausal(tiny_llama());
    auto ids = to_nntile_long(rand_ids());
    assert_module_fwd_bwd_smoke(
        *model,
        [&]() { return model->forward(ids); },
        {},
        {kB, kS, kV});
}

TEST_CASE("C++ BertMlm fwd+bwd smoke", "[models][smoke]")
{
    ContextGuard guard;
    auto model = BertMlm(tiny_bert());
    auto ids = to_nntile_long(rand_ids());
    auto tt = to_nntile_long(torch::zeros_like(ids.cpu()));
    assert_module_fwd_bwd_smoke(
        *model,
        [&]() { return model->forward(ids, tt); },
        {},
        {kB, kS, kV});
}

TEST_CASE("C++ RobertaMlm fwd+bwd smoke", "[models][smoke]")
{
    ContextGuard guard;
    auto model = RobertaMlm(tiny_roberta());
    auto ids_cpu = torch::randint(4, kV, {kB, kS}, torch::kLong);
    ids_cpu[0][0] = 1;
    auto ids = to_nntile_long(ids_cpu);
    auto tt = to_nntile_long(torch::zeros_like(ids_cpu));
    assert_module_fwd_bwd_smoke(
        *model,
        [&]() { return model->forward(ids, tt); },
        {},
        {kB, kS, kV});
}

TEST_CASE("C++ GptNeoCausal fwd+bwd smoke", "[models][smoke]")
{
    ContextGuard guard;
    auto model = GptNeoCausal(tiny_gpt_neo());
    auto ids = to_nntile_long(rand_ids());
    assert_module_fwd_bwd_smoke(
        *model,
        [&]() { return model->forward(ids); },
        {},
        {kB, kS, kV});
}

TEST_CASE("C++ GptNeoXCausal fwd+bwd smoke", "[models][smoke]")
{
    ContextGuard guard;
    auto model = GptNeoXCausal(tiny_gpt_neox());
    auto ids = to_nntile_long(rand_ids());
    assert_module_fwd_bwd_smoke(
        *model,
        [&]() { return model->forward(ids); },
        {},
        {kB, kS, kV});
}

TEST_CASE("C++ Gpt2Causal fwd+bwd smoke", "[models][smoke]")
{
    ContextGuard guard;
    auto model = Gpt2Causal(tiny_gpt2());
    auto ids = to_nntile_long(rand_ids());
    assert_module_fwd_bwd_smoke(
        *model,
        [&]() { return model->forward(ids); },
        {},
        {kB, kS, kV});
}

TEST_CASE(
    "C++ T5ForConditionalGeneration fwd+bwd smoke",
    "[models][smoke]")
{
    ContextGuard guard;
    auto model = T5ForConditionalGeneration(tiny_t5());
    auto enc = to_nntile_long(rand_ids());
    auto dec = to_nntile_long(rand_ids());
    assert_module_fwd_bwd_smoke(
        *model,
        [&]() { return model->forward(enc, dec); },
        {},
        {kB, kS, kV});
}

// ---------------------------------------------------------------------------
// Llama parts
// ---------------------------------------------------------------------------

TEST_CASE("C++ LlamaMLP fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = LlamaMLP(tiny_llama());
    auto x = to_nntile_float(rand_hidden(), true);
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ LlamaAttention fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = LlamaAttention(tiny_llama());
    auto x = to_nntile_float(rand_hidden(), true);
    at::Tensor sin;
    at::Tensor cos;
    rope_sin_cos(kB, kS, kHd, 10000.0, sin, cos);
    auto sin_n = to_nntile_float(sin, false);
    auto cos_n = to_nntile_float(cos, false);
    auto mask = bool_causal_mask(kS).contiguous().to(nntile_device());
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x, sin_n, cos_n, mask); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ LlamaDecoder fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = LlamaDecoder(tiny_llama());
    auto x = to_nntile_float(rand_hidden(), true);
    at::Tensor sin;
    at::Tensor cos;
    rope_sin_cos(kB, kS, kHd, 10000.0, sin, cos);
    auto sin_n = to_nntile_float(sin, false);
    auto cos_n = to_nntile_float(cos, false);
    auto mask = bool_causal_mask(kS).contiguous().to(nntile_device());
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x, sin_n, cos_n, mask); },
        {x},
        {kB, kS, kH});
}

// ---------------------------------------------------------------------------
// BERT parts
// ---------------------------------------------------------------------------

TEST_CASE("C++ BertSelfAttention fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = BertSelfAttention(tiny_bert());
    auto x = to_nntile_float(rand_hidden(), true);
    // After sdpa + transpose(3): [batch, seq, head_dim, heads].
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x); },
        {x},
        {kB, kS, kHd, kHeads});
}

TEST_CASE("C++ BertAttention fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = BertAttention(tiny_bert());
    auto x = to_nntile_float(rand_hidden(), true);
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ BertLayer fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = BertLayer(tiny_bert());
    auto x = to_nntile_float(rand_hidden(), true);
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ BertSelfOutput fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = BertSelfOutput(tiny_bert());
    auto heads = to_nntile_float(
        torch::randn({kB, kS, kHd, kHeads}),
        true);
    auto residual = to_nntile_float(rand_hidden(), true);
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(heads, residual); },
        {heads, residual},
        {kB, kS, kH});
}

// ---------------------------------------------------------------------------
// GPT-2 parts
// ---------------------------------------------------------------------------

TEST_CASE("C++ Gpt2Attention fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = Gpt2Attention(tiny_gpt2());
    auto x = to_nntile_float(rand_hidden(), true);
    auto mask = bool_causal_mask(kS).contiguous().to(nntile_device());
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x, mask); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ Gpt2MLP fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = Gpt2MLP(tiny_gpt2());
    auto x = to_nntile_float(rand_hidden(), true);
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ Gpt2Block fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = Gpt2Block(tiny_gpt2());
    auto x = to_nntile_float(rand_hidden(), true);
    auto mask = bool_causal_mask(kS).contiguous().to(nntile_device());
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x, mask); },
        {x},
        {kB, kS, kH});
}

// ---------------------------------------------------------------------------
// GPT-Neo parts
// ---------------------------------------------------------------------------

TEST_CASE(
    "C++ GptNeoAttention global fwd+bwd smoke",
    "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = GptNeoAttention(tiny_gpt_neo(), /*local_attn=*/false);
    auto x = to_nntile_float(rand_hidden(), true);
    auto gmask = bool_causal_mask(kS).contiguous().to(nntile_device());
    auto lmask = bool_local_causal_mask(kS, 4)
        .contiguous()
        .to(nntile_device());
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x, gmask, lmask); },
        {x},
        {kB, kS, kH});
}

TEST_CASE(
    "C++ GptNeoAttention local fwd+bwd smoke",
    "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = GptNeoAttention(tiny_gpt_neo(), /*local_attn=*/true);
    auto x = to_nntile_float(rand_hidden(), true);
    auto gmask = bool_causal_mask(kS).contiguous().to(nntile_device());
    auto lmask = bool_local_causal_mask(kS, 4)
        .contiguous()
        .to(nntile_device());
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x, gmask, lmask); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ GptNeoMLP fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = GptNeoMLP(tiny_gpt_neo());
    auto x = to_nntile_float(rand_hidden(), true);
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ GptNeoDecoder fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = GptNeoDecoder(tiny_gpt_neo(), /*local_attn=*/false);
    auto x = to_nntile_float(rand_hidden(), true);
    auto gmask = bool_causal_mask(kS).contiguous().to(nntile_device());
    auto lmask = bool_local_causal_mask(kS, 4)
        .contiguous()
        .to(nntile_device());
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x, gmask, lmask); },
        {x},
        {kB, kS, kH});
}

// ---------------------------------------------------------------------------
// GPT-NeoX parts
// ---------------------------------------------------------------------------

TEST_CASE("C++ GptNeoXMLP fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = GptNeoXMLP(tiny_gpt_neox());
    auto x = to_nntile_float(rand_hidden(), true);
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ GptNeoXAttention fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = GptNeoXAttention(tiny_gpt_neox());
    auto x = to_nntile_float(rand_hidden(), true);
    at::Tensor sin;
    at::Tensor cos;
    rope_sin_cos(kB, kS, kHd, 10000.0, sin, cos);
    auto sin_n = to_nntile_float(sin, false);
    auto cos_n = to_nntile_float(cos, false);
    auto mask = bool_causal_mask(kS).contiguous().to(nntile_device());
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x, sin_n, cos_n, mask); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ GptNeoXDecoder fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = GptNeoXDecoder(tiny_gpt_neox());
    auto x = to_nntile_float(rand_hidden(), true);
    at::Tensor sin;
    at::Tensor cos;
    rope_sin_cos(kB, kS, kHd, 10000.0, sin, cos);
    auto sin_n = to_nntile_float(sin, false);
    auto cos_n = to_nntile_float(cos, false);
    auto mask = bool_causal_mask(kS).contiguous().to(nntile_device());
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x, sin_n, cos_n, mask); },
        {x},
        {kB, kS, kH});
}

// ---------------------------------------------------------------------------
// T5 parts
// ---------------------------------------------------------------------------

TEST_CASE("C++ T5LayerFF fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = T5LayerFF(tiny_t5());
    auto x = to_nntile_float(rand_hidden(), true);
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ T5Attention self fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = T5Attention(tiny_t5(), /*cross=*/false);
    auto x = to_nntile_float(rand_hidden(), true);
    at::Tensor undef;
    auto mask = bool_causal_mask(kS).contiguous().to(nntile_device());
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x, undef, mask); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ T5Attention cross fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = T5Attention(tiny_t5(), /*cross=*/true);
    auto x = to_nntile_float(rand_hidden(), true);
    auto enc = to_nntile_float(rand_hidden(), true);
    at::Tensor undef_mask;
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x, enc, undef_mask); },
        {x, enc},
        {kB, kS, kH});
}

TEST_CASE("C++ T5EncoderBlock fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = T5EncoderBlock(tiny_t5());
    auto x = to_nntile_float(rand_hidden(), true);
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x); },
        {x},
        {kB, kS, kH});
}

TEST_CASE("C++ T5DecoderBlock fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = T5DecoderBlock(tiny_t5());
    auto x = to_nntile_float(rand_hidden(), true);
    auto enc = to_nntile_float(rand_hidden(), true);
    auto mask = bool_causal_mask(kS).contiguous().to(nntile_device());
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x, enc, mask); },
        {x, enc},
        {kB, kS, kH});
}

// ---------------------------------------------------------------------------
// MLP-Mixer parts
// ---------------------------------------------------------------------------

TEST_CASE("C++ MixerMlp side-L fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = MixerMlp('L', /*dim=*/4);
    auto x = to_nntile_float(torch::randn({8, kB, 4}), true);
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x); },
        {x},
        {8, kB, 4});
}

TEST_CASE("C++ MixerMlp side-R fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = MixerMlp('R', /*dim=*/8);
    auto x = to_nntile_float(torch::randn({8, kB, 4}), true);
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x); },
        {x},
        {8, kB, 4});
}

TEST_CASE("C++ MixerBlock fwd+bwd smoke", "[models][smoke][parts]")
{
    ContextGuard guard;
    auto mod = MixerBlock(/*channel_dim=*/8, /*patch_dim=*/4, 1e-5);
    auto x = to_nntile_float(torch::randn({8, kB, 4}), true);
    assert_module_fwd_bwd_smoke(
        *mod,
        [&]() { return mod->forward(x); },
        {x},
        {8, kB, 4});
}
