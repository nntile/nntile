#include "nntile/graph/compiled/conv2d_bwd_input_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/conv2d_bwd_input_inplace.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_conv2d_bwd_input_inplace(CompiledGraph& graph, const Conv2dAttrs& attrs,
                                  const std::string& dy_name, const std::string& c_name,
                                  const std::string& dx_name)
{
    auto& dy = graph.get_tensor<T>(dy_name);
    auto& c = graph.get_tensor<T>(c_name);
    auto& dx = graph.get_tensor<T>(dx_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::conv2d_bwd_input_inplace<T>(alpha, dy, c, beta, dx,
                                               attrs.padding, attrs.stride, attrs.dilation);
}

} // namespace

void execute_conv2d_bwd_input_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const Conv2dAttrs& attrs = std::get<Conv2dAttrs>(op_info.attrs);
    const std::string& dy_name = op_info.input_names[0];
    const std::string& c_name = op_info.input_names[1];
    const std::string& dx_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(dy_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_conv2d_bwd_input_inplace<nntile::fp32_t>(graph, attrs, dy_name, c_name, dx_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_conv2d_bwd_input_inplace<nntile::fp32_fast_tf32_t>(graph, attrs, dy_name, c_name, dx_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_conv2d_bwd_input_inplace<nntile::fp32_fast_fp16_t>(graph, attrs, dy_name, c_name, dx_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_conv2d_bwd_input_inplace<nntile::fp32_fast_bf16_t>(graph, attrs, dy_name, c_name, dx_name);
            break;
        case DataType::FP64:
            run_conv2d_bwd_input_inplace<nntile::fp64_t>(graph, attrs, dy_name, c_name, dx_name);
            break;
        case DataType::FP16:
            run_conv2d_bwd_input_inplace<nntile::fp16_t>(graph, attrs, dy_name, c_name, dx_name);
            break;
        case DataType::BF16:
            run_conv2d_bwd_input_inplace<nntile::bf16_t>(graph, attrs, dy_name, c_name, dx_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for conv2d_bwd_input_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for conv2d_bwd_input_inplace");
    }
}

} // namespace nntile::graph
