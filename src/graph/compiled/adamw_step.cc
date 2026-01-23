#include "nntile/graph/compiled/adamw_step.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/adam_step.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_adamw_step(CompiledGraph& graph, const AdamStepAttrs& attrs,
                    const std::string& grad_name, const std::string& first_moment_name,
                    const std::string& second_moment_name, const std::string& p_name)
{
    auto& grad = graph.get_tensor<T>(grad_name);
    auto& first_moment = graph.get_tensor<T>(first_moment_name);
    auto& second_moment = graph.get_tensor<T>(second_moment_name);
    auto& p = graph.get_tensor<T>(p_name);

    // For AdamW, we use the same function as Adam but the weight decay handling is different
    // The tensor operation handles this internally
    nntile::tensor::adam_step<T>(attrs.num_iter, attrs.beta_1, attrs.beta_2,
                                attrs.eps, attrs.lr, attrs.weight_decay,
                                grad, first_moment, second_moment, p);
}

} // namespace

void execute_adamw_step(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const AdamStepAttrs& attrs = std::get<AdamStepAttrs>(op_info.attrs);
    const std::string& grad_name = op_info.input_names[0];
    const std::string& first_moment_name = op_info.input_names[1];
    const std::string& second_moment_name = op_info.input_names[2];
    const std::string& p_name = op_info.input_names[3];
    DataType dtype = graph.get_dtype(grad_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_adamw_step<nntile::fp32_t>(graph, attrs, grad_name, first_moment_name, second_moment_name, p_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_adamw_step<nntile::fp32_fast_tf32_t>(graph, attrs, grad_name, first_moment_name, second_moment_name, p_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_adamw_step<nntile::fp32_fast_fp16_t>(graph, attrs, grad_name, first_moment_name, second_moment_name, p_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_adamw_step<nntile::fp32_fast_bf16_t>(graph, attrs, grad_name, first_moment_name, second_moment_name, p_name);
            break;
        case DataType::FP64:
            run_adamw_step<nntile::fp64_t>(graph, attrs, grad_name, first_moment_name, second_moment_name, p_name);
            break;
        case DataType::FP16:
            run_adamw_step<nntile::fp16_t>(graph, attrs, grad_name, first_moment_name, second_moment_name, p_name);
            break;
        case DataType::BF16:
            run_adamw_step<nntile::bf16_t>(graph, attrs, grad_name, first_moment_name, second_moment_name, p_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for adamw_step operation");
        default:
            throw std::runtime_error("Unsupported data type for adamw_step");
    }
}

} // namespace nntile::graph
