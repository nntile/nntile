/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_module.cpp
 */

#include <torch/extension.h>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "nntile_add_fiber.h"
#include "nntile_sum_slice.h"
#include "nntile_context.h"
#include "nntile_cross_entropy.h"
#include "nntile_gemm.h"
#include "nntile_mse_loss.h"
#include "nntile_rms_norm.h"
#include "nntile_rope.h"
#include "nntile_sdpa.h"
#include "nntile_transpose.h"
#include "nntile_norm.h"
#include "nntile_graph_recorder.h"
#include "nntile_sgd_step.h"
#include "nntile_adam_step.h"

#include "nntile_module_to.h"

#include <torch_nntile/models/bert.hh>
#include <torch_nntile/models/deep_relu.hh>
#include <torch_nntile/models/gpt2.hh>
#include <torch_nntile/models/gpt_neo.hh>
#include <torch_nntile/models/gpt_neox.hh>
#include <torch_nntile/models/llama.hh>
#include <torch_nntile/models/mlp_mixer.hh>
#include <torch_nntile/models/roberta.hh>
#include <torch_nntile/models/t5.hh>

#ifdef TORCH_NNTILE_USE_LIBNNTILE
#include <nntile/base_types.hh>
#endif

namespace torch_nntile
{

bool is_registered()
{
    return true;
}

bool has_libnntile()
{
#ifdef TORCH_NNTILE_USE_LIBNNTILE
    return true;
#else
    return false;
#endif
}

int64_t buffer_nbytes(const at::Tensor &tensor)
{
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        "buffer_nbytes expects an nntile tensor");
    return static_cast<int64_t>(tensor.storage().nbytes());
}

bool buffer_equal_cpu(const at::Tensor &nntile_tensor, const at::Tensor &cpu_tensor)
{
    TORCH_CHECK(
        nntile_tensor.device().type() == c10::DeviceType::PrivateUse1,
        "buffer_equal_cpu expects nntile tensor as first argument");
    TORCH_CHECK(cpu_tensor.is_cpu(), "buffer_equal_cpu expects CPU tensor");
    // Host read: .cpu() auto-compiles/runs any pending graph, then gathers.
    TORCH_CHECK(nntile_tensor.is_contiguous(), "buffer_equal_cpu: nntile tensor must be contiguous");
    at::Tensor lhs = nntile_tensor.cpu();
    at::Tensor rhs = cpu_tensor.contiguous();
    return lhs.equal(rhs);
}

void init_context_py(
    int ncpu,
    int ncuda,
    int ooc_enabled,
    const std::string &ooc_path,
    std::size_t ooc_size,
    int logger,
    int verbose,
    bool cpu_fallback)
{
    init_context(
        ncpu,
        ncuda,
        ooc_enabled,
        ooc_path.c_str(),
        ooc_size,
        logger,
        verbose,
        cpu_fallback);
}

std::vector<std::int64_t> parse_tile_sizes_py(const py::object &tile_sizes)
{
    if (py::isinstance<py::int_>(tile_sizes))
    {
        const std::int64_t value = tile_sizes.cast<std::int64_t>();
        if (value <= 0)
        {
            throw std::runtime_error(
                "torch_nntile.set_axis_group_tiling: tile size must be positive");
        }
        return {value};
    }
    if (py::isinstance<py::list>(tile_sizes) ||
        py::isinstance<py::tuple>(tile_sizes))
    {
        std::vector<std::int64_t> sizes;
        for (const py::handle item : tile_sizes)
        {
            const std::int64_t value = py::cast<std::int64_t>(item);
            if (value <= 0)
            {
                throw std::runtime_error(
                    "torch_nntile.set_axis_group_tiling: tile size must be "
                    "positive");
            }
            sizes.push_back(value);
        }
        if (sizes.empty())
        {
            throw std::runtime_error(
                "torch_nntile.set_axis_group_tiling: tile_sizes must be "
                "non-empty");
        }
        return sizes;
    }
    throw std::runtime_error(
        "torch_nntile.set_axis_group_tiling: tile_sizes must be int or "
        "sequence of ints");
}

void set_axis_group_name_py(
    const at::Tensor &tensor,
    const py::dict &names)
{
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        "set_axis_group_name expects an nntile tensor");
    std::unordered_map<int, std::string> parsed;
    for (const auto &item : names)
    {
        const int dim = py::cast<int>(item.first);
        const std::string name = py::cast<std::string>(item.second);
        parsed.emplace(dim, name);
    }
    set_axis_group_name(tensor, parsed);
    stage_tensor_for_axis_group_compile(tensor);
}

void set_axis_group_tiling_py(
    const std::string &name,
    const py::object &tile_sizes)
{
    set_axis_group_tiling(name, parse_tile_sizes_py(tile_sizes));
}

} // namespace torch_nntile

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("is_registered", &torch_nntile::is_registered, "Backend loaded");
    m.def(
        "has_libnntile",
        &torch_nntile::has_libnntile,
        "Whether libnntile TensorGraph add is linked");
    m.def(
        "built_with_cuda",
        &torch_nntile::built_with_cuda,
        "Whether linked libnntile was built with CUDA");
    m.def("buffer_nbytes", &torch_nntile::buffer_nbytes, "Storage nbytes");
    m.def(
        "buffer_equal_cpu",
        &torch_nntile::buffer_equal_cpu,
        "Compare nntile tensor to CPU tensor");
    m.def(
        "init_context",
        &torch_nntile::init_context_py,
        "Configure StarPU workers before the first nntile op",
        py::arg("ncpu") = -1,
        py::arg("ncuda") = -1,
        py::arg("ooc_enabled") = 0,
        py::arg("ooc_path") = "/tmp/nntile_ooc",
        py::arg("ooc_size") = 16 * 1024 * 1024,
        py::arg("logger") = 0,
        py::arg("verbose") = 0,
        py::arg("cpu_fallback") = true);
    m.def(
        "is_cpu_fallback_enabled",
        &torch_nntile::is_cpu_fallback_enabled,
        "Whether unsupported ops may fall back to CPU");
    m.def(
        "is_context_initialized",
        &torch_nntile::is_context_initialized,
        "Whether the libnntile context has been created");
    m.def(
        "restrict_cpu",
        &torch_nntile::restrict_cpu,
        "Run StarPU codelets on CPU workers only");
    m.def(
        "restrict_cuda",
        &torch_nntile::restrict_cuda,
        "Run StarPU codelets on CUDA workers only");
    m.def(
        "restore_where",
        &torch_nntile::restore_where,
        "Restore default StarPU codelet worker placement");
    m.def(
        "wait_for_all",
        &torch_nntile::wait_for_all,
        "Block until all submitted StarPU tasks finish");
    m.def(
        "shutdown_context",
        &torch_nntile::shutdown_context,
        "Shut down libnntile / StarPU (safe to call repeatedly)");
    m.def(
        "execute",
        &torch_nntile::execute_pending_graph,
        "Compile and submit the pending TensorGraph (does not wait; call wait())");
    m.def(
        "compile_graph",
        &torch_nntile::compile_graph,
        "Lower and compile the pending TensorGraph into a persistent session");
    m.def(
        "run",
        &torch_nntile::run_graph,
        "Submit the compiled graph to StarPU (asynchronous; does not wait)");
    m.def(
        "reset_graph_session",
        &torch_nntile::reset_graph_session,
        "Discard the compiled graph session and recorder state");
    m.def(
        "has_pending_graph",
        &torch_nntile::has_pending_graph,
        "Whether a deferred TensorGraph is waiting for compile/run");
    m.def(
        "set_axis_group_name",
        &torch_nntile::set_axis_group_name_py,
        "Name TensorGraph axis groups for selected tensor dimensions",
        py::arg("tensor"),
        py::arg("names"));
    m.def(
        "set_axis_group_tiling",
        &torch_nntile::set_axis_group_tiling_py,
        "Set tiling for a named axis group before compile/run",
        py::arg("name"),
        py::arg("tile_sizes"));
    m.def(
        "format_axis_groups",
        &torch_nntile::format_axis_groups,
        "Format pending TensorGraph axis groups (like C++ TensorGraph::to_string)");
    m.def(
        "print_axis_groups",
        &torch_nntile::print_axis_groups,
        "Print pending TensorGraph axis groups to stdout");
    m.def(
        "print_info",
        &torch_nntile::print_info,
        "Print cumulative compile/run/wait/host-readout timing stats");
    m.def(
        "add_fiber_forward",
        &torch_nntile::add_fiber_forward,
        "NNTile add_fiber forward (no broadcast expand)",
        py::arg("fiber"),
        py::arg("tensor"),
        py::arg("axis"),
        py::arg("batch_ndim"),
        py::arg("alpha") = 1.0,
        py::arg("beta") = 1.0);
    m.def(
        "add_fiber_backward",
        &torch_nntile::add_fiber_backward,
        "NNTile add_fiber backward (sum_fiber for fiber grad)",
        py::arg("grad_out"),
        py::arg("fiber"),
        py::arg("tensor"),
        py::arg("axis"),
        py::arg("batch_ndim"),
        py::arg("output_mask"),
        py::arg("alpha") = 1.0,
        py::arg("beta") = 1.0);
    m.def(
        "sum_slice_forward",
        &torch_nntile::sum_slice_forward,
        "NNTile sum_slice forward (GAP reduction)",
        py::arg("src"),
        py::arg("axis"),
        py::arg("alpha") = 1.0,
        py::arg("beta") = 0.0);
    m.def(
        "sum_slice_backward",
        &torch_nntile::sum_slice_backward,
        "NNTile sum_slice backward (add_slice broadcast)",
        py::arg("grad_out"),
        py::arg("src"),
        py::arg("axis"),
        py::arg("alpha") = 1.0);
    m.def(
        "gemm_forward",
        &torch_nntile::gemm_forward,
        "NNTile GEMM forward (N-D contraction, C++ graph API semantics)",
        py::arg("a"),
        py::arg("b"),
        py::arg("ndim"),
        py::arg("batch_ndim") = 0,
        py::arg("trans_a") = false,
        py::arg("trans_b") = false);
    m.def(
        "gemm_backward",
        &torch_nntile::gemm_backward,
        "NNTile GEMM backward",
        py::arg("a"),
        py::arg("b"),
        py::arg("grad_out"),
        py::arg("ndim"),
        py::arg("batch_ndim"),
        py::arg("output_mask"),
        py::arg("trans_a") = false,
        py::arg("trans_b") = false);
    m.def(
        "cross_entropy_forward",
        &torch_nntile::cross_entropy_forward,
        "NNTile cross-entropy forward; returns (loss, maxsumexp)",
        py::arg("logits"),
        py::arg("target"),
        py::arg("reduction") = 1,
        py::arg("ignore_index") = -100);
    m.def(
        "cross_entropy_backward",
        &torch_nntile::cross_entropy_backward,
        "NNTile cross-entropy backward w.r.t. logits (reuses maxsumexp)",
        py::arg("logits"),
        py::arg("target"),
        py::arg("grad_output"),
        py::arg("maxsumexp"),
        py::arg("reduction") = 1,
        py::arg("ignore_index") = -100);
    m.def(
        "sgd_step",
        &torch_nntile::sgd_step,
        "Fused SGD step on nntile tensors (updates param and velocity in-place)",
        py::arg("param"),
        py::arg("grad"),
        py::arg("velocity"),
        py::arg("num_iter"),
        py::arg("lr"),
        py::arg("momentum") = 0.0,
        py::arg("weight_decay") = 0.0,
        py::arg("dampening") = 0.0,
        py::arg("nesterov") = false);
    m.def(
        "adam_step",
        &torch_nntile::adam_step,
        "Fused Adam step on nntile tensors (updates param and moments in-place)",
        py::arg("param"),
        py::arg("grad"),
        py::arg("first_moment"),
        py::arg("second_moment"),
        py::arg("num_iter"),
        py::arg("lr"),
        py::arg("beta_1") = 0.9,
        py::arg("beta_2") = 0.999,
        py::arg("eps") = 1e-8,
        py::arg("weight_decay") = 0.0);
    m.def(
        "adamw_step",
        &torch_nntile::adamw_step,
        "Fused AdamW step on nntile tensors (updates param and moments in-place)",
        py::arg("param"),
        py::arg("grad"),
        py::arg("first_moment"),
        py::arg("second_moment"),
        py::arg("num_iter"),
        py::arg("lr"),
        py::arg("beta_1") = 0.9,
        py::arg("beta_2") = 0.999,
        py::arg("eps") = 1e-8,
        py::arg("weight_decay") = 0.0);
    m.def(
        "rms_norm_forward",
        &torch_nntile::rms_norm_forward,
        "NNTile RMSNorm forward",
        py::arg("input"),
        py::arg("normalized_shape"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none());
    m.def(
        "rms_norm_backward",
        &torch_nntile::rms_norm_backward,
        "NNTile RMSNorm backward",
        py::arg("grad_out"),
        py::arg("input"),
        py::arg("normalized_shape"),
        py::arg("rstd"),
        py::arg("weight") = py::none(),
        py::arg("output_mask"));
    m.def(
        "rope_forward",
        &torch_nntile::rope_forward,
        "NNTile RoPE forward",
        py::arg("sin"),
        py::arg("cos"),
        py::arg("x"));
    m.def(
        "rope_backward",
        &torch_nntile::rope_backward,
        "NNTile RoPE backward (grad w.r.t. x)",
        py::arg("sin"),
        py::arg("cos"),
        py::arg("grad_out"),
        py::arg("output_mask"));
    m.def(
        "mse_loss_forward",
        &torch_nntile::mse_loss_forward,
        "NNTile MSE loss forward: scale * ||x||^2",
        py::arg("x"),
        py::arg("scale") = 1.0);
    m.def(
        "mse_loss_backward",
        &torch_nntile::mse_loss_backward,
        "NNTile MSE loss backward: grad_x = 2*scale*x",
        py::arg("x"),
        py::arg("scale"),
        py::arg("needs_grad"));
    m.def(
        "sdpa_forward",
        &torch_nntile::sdpa_forward,
        "NNTile SDPA eager forward (NNTile tensor layout)",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("batch_ndim") = 2);
    m.def(
        "sdpa_backward",
        &torch_nntile::sdpa_backward,
        "NNTile SDPA eager backward",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("grad_out"),
        py::arg("mask") = py::none(),
        py::arg("batch_ndim") = 2);
    m.def(
        "model_transpose_forward",
        &torch_nntile::model_transpose_forward,
        "NNTile model-code transpose forward",
        py::arg("x"),
        py::arg("model_ndim"));
    m.def(
        "model_transpose_backward",
        &torch_nntile::model_transpose_backward,
        "NNTile model-code transpose backward",
        py::arg("grad_out"),
        py::arg("model_ndim"),
        py::arg("x"));
    m.def(
        "norm_forward",
        [](const at::Tensor &input,
           std::optional<int64_t> dim,
           bool keepdim,
           std::optional<at::Tensor> out) {
            at::Tensor *out_ptr = nullptr;
            at::Tensor out_tensor;
            if (out.has_value())
            {
                out_tensor = *out;
                out_ptr = &out_tensor;
            }
            return torch_nntile::norm_forward(
                input,
                dim,
                keepdim,
                out_ptr);
        },
        "NNTile 2-norm forward",
        py::arg("input"),
        py::arg("dim") = py::none(),
        py::arg("keepdim") = false,
        py::arg("out") = py::none());

    // C++ libtorch_nntile models (NNGraph ports) - tested from Python.
    m.def(
        "cpp_models_listed",
        []() {
            return std::vector<std::string>{
                "DeepReLU",
                "Gpt2Causal",
                "GptNeoCausal",
                "GptNeoXCausal",
                "LlamaCausal",
                "BertMlm",
                "RobertaMlm",
                "T5",
                "MlpMixer",
            };
        },
        "Names of C++ torch::nn models in libtorch_nntile");
    m.def(
        "cpp_llama_causal_forward",
        [](const at::Tensor &input_ids,
           int64_t vocab_size,
           int64_t hidden_size,
           int64_t intermediate_size,
           int64_t num_hidden_layers,
           int64_t num_attention_heads,
           int64_t num_key_value_heads) {
            using torch_nntile::models::LlamaCausal;
            using torch_nntile::models::LlamaConfig;
            LlamaConfig cfg;
            cfg.vocab_size = vocab_size;
            cfg.hidden_size = hidden_size;
            cfg.intermediate_size = intermediate_size;
            cfg.num_hidden_layers = num_hidden_layers;
            cfg.num_attention_heads = num_attention_heads;
            cfg.num_key_value_heads = num_key_value_heads;
            cfg.max_position_embeddings =
                std::max<int64_t>(input_ids.size(1), 8);
            auto model = LlamaCausal(cfg);
            torch_nntile::module_to_device(*model, input_ids.device());
            model->warm_rope_cache(
                input_ids.size(0),
                input_ids.size(1),
                input_ids.device());
            return model->forward(input_ids);
        },
        "Run C++ LlamaCausal forward (device follows input_ids)",
        py::arg("input_ids"),
        py::arg("vocab_size") = 128,
        py::arg("hidden_size") = 64,
        py::arg("intermediate_size") = 128,
        py::arg("num_hidden_layers") = 1,
        py::arg("num_attention_heads") = 4,
        py::arg("num_key_value_heads") = 4);
    m.def(
        "cpp_bert_mlm_forward",
        [](const at::Tensor &input_ids,
           const at::Tensor &token_type_ids,
           int64_t vocab_size,
           int64_t hidden_size,
           int64_t intermediate_size,
           int64_t num_hidden_layers,
           int64_t num_attention_heads) {
            using torch_nntile::models::BertConfig;
            using torch_nntile::models::BertMlm;
            BertConfig cfg;
            cfg.vocab_size = vocab_size;
            cfg.hidden_size = hidden_size;
            cfg.intermediate_size = intermediate_size;
            cfg.num_hidden_layers = num_hidden_layers;
            cfg.num_attention_heads = num_attention_heads;
            cfg.max_position_embeddings =
                std::max<int64_t>(input_ids.size(1), 8);
            auto model = BertMlm(cfg);
            torch_nntile::module_to_device(*model, input_ids.device());
            return model->forward(input_ids, token_type_ids);
        },
        "Run C++ BertMlm forward",
        py::arg("input_ids"),
        py::arg("token_type_ids"),
        py::arg("vocab_size") = 128,
        py::arg("hidden_size") = 64,
        py::arg("intermediate_size") = 128,
        py::arg("num_hidden_layers") = 1,
        py::arg("num_attention_heads") = 4);
    m.def(
        "cpp_roberta_mlm_forward",
        [](const at::Tensor &input_ids,
           const at::Tensor &token_type_ids,
           int64_t vocab_size,
           int64_t hidden_size,
           int64_t intermediate_size,
           int64_t num_hidden_layers,
           int64_t num_attention_heads,
           int64_t pad_token_id) {
            using torch_nntile::models::RobertaConfig;
            using torch_nntile::models::RobertaMlm;
            RobertaConfig cfg;
            cfg.vocab_size = vocab_size;
            cfg.hidden_size = hidden_size;
            cfg.intermediate_size = intermediate_size;
            cfg.num_hidden_layers = num_hidden_layers;
            cfg.num_attention_heads = num_attention_heads;
            cfg.pad_token_id = pad_token_id;
            cfg.max_position_embeddings =
                std::max<int64_t>(input_ids.size(1) + 2, 16);
            auto model = RobertaMlm(cfg);
            torch_nntile::module_to_device(*model, input_ids.device());
            return model->forward(input_ids, token_type_ids);
        },
        "Run C++ RobertaMlm forward (pad-aware positions)",
        py::arg("input_ids"),
        py::arg("token_type_ids"),
        py::arg("vocab_size") = 128,
        py::arg("hidden_size") = 64,
        py::arg("intermediate_size") = 128,
        py::arg("num_hidden_layers") = 1,
        py::arg("num_attention_heads") = 4,
        py::arg("pad_token_id") = 1);
    m.def(
        "cpp_gpt_neo_causal_forward",
        [](const at::Tensor &input_ids,
           int64_t vocab_size,
           int64_t hidden_size,
           int64_t intermediate_size,
           int64_t num_hidden_layers,
           int64_t num_attention_heads,
           int64_t window_size) {
            using torch_nntile::models::GptNeoCausal;
            using torch_nntile::models::GptNeoConfig;
            GptNeoConfig cfg;
            cfg.vocab_size = vocab_size;
            cfg.hidden_size = hidden_size;
            cfg.intermediate_size = intermediate_size;
            cfg.num_hidden_layers = num_hidden_layers;
            cfg.num_attention_heads = num_attention_heads;
            cfg.window_size = window_size;
            cfg.max_position_embeddings =
                std::max<int64_t>(input_ids.size(1), 8);
            cfg.attention_layers.clear();
            for (int64_t i = 0; i < num_hidden_layers; ++i)
            {
                cfg.attention_layers.push_back(
                    (i % 2 == 1) ? "local" : "global");
            }
            auto model = GptNeoCausal(cfg);
            torch_nntile::module_to_device(*model, input_ids.device());
            return model->forward(input_ids);
        },
        "Run C++ GptNeoCausal forward",
        py::arg("input_ids"),
        py::arg("vocab_size") = 128,
        py::arg("hidden_size") = 64,
        py::arg("intermediate_size") = 128,
        py::arg("num_hidden_layers") = 2,
        py::arg("num_attention_heads") = 4,
        py::arg("window_size") = 4);
    m.def(
        "cpp_gpt_neox_causal_forward",
        [](const at::Tensor &input_ids,
           int64_t vocab_size,
           int64_t hidden_size,
           int64_t intermediate_size,
           int64_t num_hidden_layers,
           int64_t num_attention_heads,
           double rotary_pct) {
            using torch_nntile::models::GptNeoXCausal;
            using torch_nntile::models::GptNeoXConfig;
            GptNeoXConfig cfg;
            cfg.vocab_size = vocab_size;
            cfg.hidden_size = hidden_size;
            cfg.intermediate_size = intermediate_size;
            cfg.num_hidden_layers = num_hidden_layers;
            cfg.num_attention_heads = num_attention_heads;
            cfg.rotary_pct = rotary_pct;
            cfg.max_position_embeddings =
                std::max<int64_t>(input_ids.size(1), 8);
            auto model = GptNeoXCausal(cfg);
            torch_nntile::module_to_device(*model, input_ids.device());
            model->warm_rope_cache(
                input_ids.size(0),
                input_ids.size(1),
                input_ids.device());
            return model->forward(input_ids);
        },
        "Run C++ GptNeoXCausal forward",
        py::arg("input_ids"),
        py::arg("vocab_size") = 128,
        py::arg("hidden_size") = 64,
        py::arg("intermediate_size") = 128,
        py::arg("num_hidden_layers") = 1,
        py::arg("num_attention_heads") = 4,
        py::arg("rotary_pct") = 0.25);
    m.def(
        "cpp_gpt2_causal_forward",
        [](const at::Tensor &input_ids,
           int64_t vocab_size,
           int64_t n_embd,
           int64_t n_head,
           int64_t n_layer) {
            using torch_nntile::models::Gpt2Causal;
            using torch_nntile::models::Gpt2Config;
            Gpt2Config cfg;
            cfg.vocab_size = vocab_size;
            cfg.n_embd = n_embd;
            cfg.n_head = n_head;
            cfg.n_layer = n_layer;
            cfg.n_positions = std::max<int64_t>(input_ids.size(1), 8);
            auto model = Gpt2Causal(cfg);
            torch_nntile::module_to_device(*model, input_ids.device());
            model->warm_sequence_cache(
                input_ids.size(0),
                input_ids.size(1),
                input_ids.device());
            return model->forward(input_ids);
        },
        "Run C++ Gpt2Causal forward",
        py::arg("input_ids"),
        py::arg("vocab_size") = 128,
        py::arg("n_embd") = 64,
        py::arg("n_head") = 4,
        py::arg("n_layer") = 1);
    m.def(
        "cpp_t5_forward",
        [](const at::Tensor &encoder_ids,
           const at::Tensor &decoder_ids,
           int64_t vocab_size,
           int64_t d_model,
           int64_t d_kv,
           int64_t d_ff,
           int64_t num_layers,
           int64_t num_heads) {
            using torch_nntile::models::T5Config;
            using torch_nntile::models::T5ForConditionalGeneration;
            T5Config cfg;
            cfg.vocab_size = vocab_size;
            cfg.d_model = d_model;
            cfg.d_kv = d_kv;
            cfg.d_ff = d_ff;
            cfg.num_layers = num_layers;
            cfg.num_decoder_layers = num_layers;
            cfg.num_heads = num_heads;
            auto model = T5ForConditionalGeneration(cfg);
            torch_nntile::module_to_device(*model, encoder_ids.device());
            return model->forward(encoder_ids, decoder_ids);
        },
        "Run C++ T5ForConditionalGeneration forward",
        py::arg("encoder_ids"),
        py::arg("decoder_ids"),
        py::arg("vocab_size") = 128,
        py::arg("d_model") = 64,
        py::arg("d_kv") = 16,
        py::arg("d_ff") = 128,
        py::arg("num_layers") = 1,
        py::arg("num_heads") = 4);
    m.def(
        "cpp_mlp_mixer_forward",
        [](const at::Tensor &x,
           int64_t channel_dim,
           int64_t init_patch_dim,
           int64_t projected_patch_dim,
           int64_t num_mixer_layers,
           int64_t n_classes) {
            using torch_nntile::models::MlpMixer;
            using torch_nntile::models::MlpMixerConfig;
            MlpMixerConfig cfg;
            cfg.channel_dim = channel_dim;
            cfg.init_patch_dim = init_patch_dim;
            cfg.projected_patch_dim = projected_patch_dim;
            cfg.num_mixer_layers = num_mixer_layers;
            cfg.n_classes = n_classes;
            auto model = MlpMixer(cfg);
            torch_nntile::module_to_device(*model, x.device());
            return model->forward(x);
        },
        "Run C++ MlpMixer forward (device follows x)",
        py::arg("x"),
        py::arg("channel_dim") = 8,
        py::arg("init_patch_dim") = 4,
        py::arg("projected_patch_dim") = 4,
        py::arg("num_mixer_layers") = 2,
        py::arg("n_classes") = 3);
}
