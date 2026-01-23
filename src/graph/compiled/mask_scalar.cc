#include "nntile/graph/compiled/mask_scalar.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/mask_scalar.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_mask_scalar(CompiledGraph& graph, const MaskScalarAttrs& attrs,
                     const std::string& mask_name, const std::string& x_name)
{
    auto& mask = graph.get_tensor<bool_t>(mask_name);
    auto& x = graph.get_tensor<T>(x_name);

    const auto val = static_cast<nntile::Scalar>(attrs.val);

    nntile::tensor::mask_scalar<T>(mask, val, x, attrs.batch_ndim);
}

} // namespace

void execute_mask_scalar(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const MaskScalarAttrs& attrs = std::get<MaskScalarAttrs>(op_info.attrs);
    const std::string& mask_name = op_info.input_names[0];
    const std::string& x_name = op_info.input_names[1];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_mask_scalar<nntile::fp32_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_mask_scalar<nntile::fp32_fast_tf32_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_mask_scalar<nntile::fp32_fast_fp16_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_mask_scalar<nntile::fp32_fast_bf16_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::FP64:
            run_mask_scalar<nntile::fp64_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::FP16:
            run_mask_scalar<nntile::fp16_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::BF16:
            run_mask_scalar<nntile::bf16_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for mask_scalar operation");
        default:
            throw std::runtime_error("Unsupported data type for mask_scalar");
    }
}

} // namespace nntile::graph
