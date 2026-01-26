#include "nntile/graph/compiled/rope.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/rope.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_rope(CompiledGraph& graph,
              const std::string& sin_name, const std::string& cos_name,
              const std::string& src_name, const std::string& dst_name)
{
    auto& sin_tensor = graph.get_tensor<T>(sin_name);
    auto& cos_tensor = graph.get_tensor<T>(cos_name);
    auto& src = graph.get_tensor<T>(src_name);
    auto& dst = graph.get_tensor<T>(dst_name);

    nntile::tensor::rope<T>(sin_tensor, cos_tensor, src, dst);
}

} // namespace

void execute_rope(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& sin_name = op_info.input_names[0];
    const std::string& cos_name = op_info.input_names[1];
    const std::string& src_name = op_info.input_names[2];
    const std::string& dst_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(sin_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_rope<nntile::fp32_t>(graph, sin_name, cos_name, src_name, dst_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_rope<nntile::fp32_fast_tf32_t>(graph, sin_name, cos_name, src_name, dst_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_rope<nntile::fp32_fast_fp16_t>(graph, sin_name, cos_name, src_name, dst_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_rope<nntile::fp32_fast_bf16_t>(graph, sin_name, cos_name, src_name, dst_name);
            break;
        case DataType::FP64:
            run_rope<nntile::fp64_t>(graph, sin_name, cos_name, src_name, dst_name);
            break;
        case DataType::FP16:
            run_rope<nntile::fp16_t>(graph, sin_name, cos_name, src_name, dst_name);
            break;
        case DataType::BF16:
            run_rope<nntile::bf16_t>(graph, sin_name, cos_name, src_name, dst_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for rope operation");
        default:
            throw std::runtime_error("Unsupported data type for rope");
    }
}

} // namespace nntile::graph
