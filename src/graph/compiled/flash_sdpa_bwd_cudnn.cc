#include "nntile/graph/compiled/flash_sdpa_bwd_cudnn.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/flash_sdpa_bwd_cudnn.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_flash_sdpa_bwd_cudnn(CompiledGraph& graph, const ClearAttrs& attrs,
                              const std::string& K_name, const std::string& Q_name,
                              const std::string& V_name, const std::string& A_name,
                              const std::string& dA_name, const std::string& mask_name,
                              const std::string& logsumexp_name, const std::string& dK_name,
                              const std::string& dQ_name, const std::string& dV_name)
{
    auto& K = graph.get_tensor<T>(K_name);
    auto& Q = graph.get_tensor<T>(Q_name);
    auto& V = graph.get_tensor<T>(V_name);
    auto& A = graph.get_tensor<T>(A_name);
    auto& dA = graph.get_tensor<T>(dA_name);
    auto& mask = graph.get_tensor<T>(mask_name);
    auto& logsumexp = graph.get_tensor<fp32_t>(logsumexp_name);
    auto& dK = graph.get_tensor<T>(dK_name);
    auto& dQ = graph.get_tensor<T>(dQ_name);
    auto& dV = graph.get_tensor<T>(dV_name);

    nntile::tensor::flash_sdpa_bwd_cudnn<T>(K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);
}

} // namespace

void execute_flash_sdpa_bwd_cudnn(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ClearAttrs& attrs = std::get<ClearAttrs>(op_info.attrs);
    const std::string& K_name = op_info.input_names[0];
    const std::string& Q_name = op_info.input_names[1];
    const std::string& V_name = op_info.input_names[2];
    const std::string& A_name = op_info.input_names[3];
    const std::string& dA_name = op_info.input_names[4];
    const std::string& mask_name = op_info.input_names[5];
    const std::string& logsumexp_name = op_info.input_names[6];
    const std::string& dK_name = op_info.output_names[0];
    const std::string& dQ_name = op_info.output_names[1];
    const std::string& dV_name = op_info.output_names[2];
    DataType dtype = graph.get_dtype(K_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_flash_sdpa_bwd_cudnn<nntile::fp32_t>(graph, attrs, K_name, Q_name, V_name, A_name, dA_name, mask_name, logsumexp_name, dK_name, dQ_name, dV_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_flash_sdpa_bwd_cudnn<nntile::fp32_fast_tf32_t>(graph, attrs, K_name, Q_name, V_name, A_name, dA_name, mask_name, logsumexp_name, dK_name, dQ_name, dV_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_flash_sdpa_bwd_cudnn<nntile::fp32_fast_fp16_t>(graph, attrs, K_name, Q_name, V_name, A_name, dA_name, mask_name, logsumexp_name, dK_name, dQ_name, dV_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_flash_sdpa_bwd_cudnn<nntile::fp32_fast_bf16_t>(graph, attrs, K_name, Q_name, V_name, A_name, dA_name, mask_name, logsumexp_name, dK_name, dQ_name, dV_name);
            break;
        case DataType::FP64:
            run_flash_sdpa_bwd_cudnn<nntile::fp64_t>(graph, attrs, K_name, Q_name, V_name, A_name, dA_name, mask_name, logsumexp_name, dK_name, dQ_name, dV_name);
            break;
        case DataType::FP16:
            run_flash_sdpa_bwd_cudnn<nntile::fp16_t>(graph, attrs, K_name, Q_name, V_name, A_name, dA_name, mask_name, logsumexp_name, dK_name, dQ_name, dV_name);
            break;
        case DataType::BF16:
            run_flash_sdpa_bwd_cudnn<nntile::bf16_t>(graph, attrs, K_name, Q_name, V_name, A_name, dA_name, mask_name, logsumexp_name, dK_name, dQ_name, dV_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for flash_sdpa_bwd_cudnn operation");
        default:
            throw std::runtime_error("Unsupported data type for flash_sdpa_bwd_cudnn");
    }
}

} // namespace nntile::graph
