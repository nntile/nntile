#include "nntile/graph/compiled/sqrt_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/sqrt_inplace.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_sqrt_inplace(CompiledGraph& graph, const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    nntile::tensor::sqrt_inplace<T>(x);
}

} // namespace

void execute_sqrt_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_sqrt_inplace<nntile::fp32_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_sqrt_inplace<nntile::fp32_fast_tf32_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_sqrt_inplace<nntile::fp32_fast_fp16_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_sqrt_inplace<nntile::fp32_fast_bf16_t>(graph, x_name);
            break;
        case DataType::FP64:
            run_sqrt_inplace<nntile::fp64_t>(graph, x_name);
            break;
        case DataType::FP16:
            run_sqrt_inplace<nntile::fp16_t>(graph, x_name);
            break;
        case DataType::BF16:
            run_sqrt_inplace<nntile::bf16_t>(graph, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sqrt_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for sqrt_inplace");
    }
}

} // namespace nntile::graph
