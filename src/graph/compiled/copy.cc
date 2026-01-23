#include "nntile/graph/compiled/copy.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/copy.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_copy(CompiledGraph& graph, const ClearAttrs& attrs,
              const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    nntile::tensor::copy<T>(x, y);
}

} // namespace

void execute_copy(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ClearAttrs& attrs = std::get<ClearAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_copy<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_copy<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_copy<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_copy<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_copy<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_copy<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_copy<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for copy operation");
        default:
            throw std::runtime_error("Unsupported data type for copy");
    }
}

} // namespace nntile::graph
