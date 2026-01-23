#include "nntile/graph/compiled/sgd_step.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/sgd_step.hh"

namespace nntile::graph
{

namespace
{

// Optimizer operations
template<typename T>
void run_sgd_step(CompiledGraph& graph, const SgdStepAttrs& attrs,
                  const std::string& grad_name, const std::string& velocity_name,
                  const std::string& p_name)
{
    auto& grad = graph.get_tensor<T>(grad_name);
    auto& velocity = graph.get_tensor<T>(velocity_name);
    auto& p = graph.get_tensor<T>(p_name);

    nntile::tensor::sgd_step<T>(attrs.num_iter, attrs.momentum, attrs.lr,
                               attrs.weight_decay, attrs.dampening, attrs.nesterov,
                               grad, velocity, p);
}

} // namespace

void execute_sgd_step(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const SgdStepAttrs& attrs = std::get<SgdStepAttrs>(op_info.attrs);
    const std::string& grad_name = op_info.input_names[0];
    const std::string& velocity_name = op_info.input_names[1];
    const std::string& p_name = op_info.input_names[2];
    DataType dtype = graph.get_dtype(grad_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_sgd_step<nntile::fp32_t>(graph, attrs, grad_name, velocity_name, p_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_sgd_step<nntile::fp32_fast_tf32_t>(graph, attrs, grad_name, velocity_name, p_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_sgd_step<nntile::fp32_fast_fp16_t>(graph, attrs, grad_name, velocity_name, p_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_sgd_step<nntile::fp32_fast_bf16_t>(graph, attrs, grad_name, velocity_name, p_name);
            break;
        case DataType::FP64:
            run_sgd_step<nntile::fp64_t>(graph, attrs, grad_name, velocity_name, p_name);
            break;
        case DataType::FP16:
            run_sgd_step<nntile::fp16_t>(graph, attrs, grad_name, velocity_name, p_name);
            break;
        case DataType::BF16:
            run_sgd_step<nntile::bf16_t>(graph, attrs, grad_name, velocity_name, p_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sgd_step operation");
        default:
            throw std::runtime_error("Unsupported data type for sgd_step");
    }
}

} // namespace nntile::graph
