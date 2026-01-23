#include "nntile/graph/compiled/subtract_indexed_outputs.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/subtract_indexed_outputs.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_subtract_indexed_outputs(CompiledGraph& graph, const SubtractIndexedOutputsAttrs& attrs,
                                  const std::string& labels_name, const std::string& x_name)
{
    auto& labels = graph.get_tensor<int64_t>(labels_name);
    auto& x = graph.get_tensor<T>(x_name);

    nntile::tensor::subtract_indexed_outputs<T>(attrs.val, labels, x, attrs.ignore_index);
}

} // namespace

void execute_subtract_indexed_outputs(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const SubtractIndexedOutputsAttrs& attrs = std::get<SubtractIndexedOutputsAttrs>(op_info.attrs);
    const std::string& labels_name = op_info.input_names[0];
    const std::string& x_name = op_info.input_names[1];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_subtract_indexed_outputs<nntile::fp32_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_subtract_indexed_outputs<nntile::fp32_fast_tf32_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_subtract_indexed_outputs<nntile::fp32_fast_fp16_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_subtract_indexed_outputs<nntile::fp32_fast_bf16_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::FP64:
            run_subtract_indexed_outputs<nntile::fp64_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::FP16:
            run_subtract_indexed_outputs<nntile::fp16_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::BF16:
            run_subtract_indexed_outputs<nntile::bf16_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for subtract_indexed_outputs operation");
        default:
            throw std::runtime_error("Unsupported data type for subtract_indexed_outputs");
    }
}

} // namespace nntile::graph
