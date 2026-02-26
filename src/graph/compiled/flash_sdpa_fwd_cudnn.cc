#include "nntile/graph/compiled/flash_sdpa_fwd_cudnn.hh"

#include <stdexcept>

#include <starpu.h>

#include "nntile/base_types.hh"
#include "nntile/tensor/flash_sdpa_fwd_cudnn.hh"

namespace nntile::graph
{

namespace
{

// Advanced operations (CUDA-only for flash attention)
template<typename T>
void run_flash_sdpa_fwd_cudnn(CompiledGraph& graph, const ClearAttrs& attrs,
                              const std::string& K_name, const std::string& Q_name,
                              const std::string& mask_name, const std::string& logsumexp_name,
                              const std::string& V_name, const std::string& A_name)
{
    auto& K = graph.get_tensor<T>(K_name);
    auto& Q = graph.get_tensor<T>(Q_name);
    auto& mask = graph.get_tensor<T>(mask_name);
    auto& logsumexp = graph.get_tensor<fp32_t>(logsumexp_name);
    auto& V = graph.get_tensor<T>(V_name);
    auto& A = graph.get_tensor<T>(A_name);

    nntile::tensor::flash_sdpa_fwd_cudnn<T>(K, Q, mask, logsumexp, V, A);
}

} // namespace

void execute_flash_sdpa_fwd_cudnn(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    // Flash attention requires CUDA
    if(!starpu_cuda_worker_get_count())
    {
        throw std::runtime_error("flash_sdpa_fwd_cudnn operation requires CUDA but no CUDA workers are available");
    }

    const ClearAttrs& attrs = std::get<ClearAttrs>(op_info.attrs);
    const std::string& K_name = op_info.input_names[0];
    const std::string& Q_name = op_info.input_names[1];
    const std::string& mask_name = op_info.input_names[2];
    const std::string& logsumexp_name = op_info.input_names[3];
    const std::string& V_name = op_info.input_names[4];
    const std::string& A_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(K_name);

    switch(dtype)
    {
        case DataType::FP32:
            throw std::runtime_error("FP32 data type not supported for flash_sdpa_fwd_cudnn operation");
            break;
        case DataType::FP32_FAST_TF32:
            throw std::runtime_error("FP32_FAST_TF32 data type not supported for flash_sdpa_fwd_cudnn operation");
            break;
        case DataType::FP32_FAST_FP16:
            throw std::runtime_error("FP32_FAST_FP16 data type not supported for flash_sdpa_fwd_cudnn operation");
            break;
        case DataType::FP32_FAST_BF16:
            throw std::runtime_error("FP32_FAST_BF16 data type not supported for flash_sdpa_fwd_cudnn operation");
            break;
        case DataType::FP64:
            throw std::runtime_error("FP64 data type not supported for flash_sdpa_fwd_cudnn operation");
            break;
        case DataType::FP16:
            run_flash_sdpa_fwd_cudnn<nntile::fp16_t>(graph, attrs, K_name, Q_name, mask_name, logsumexp_name, V_name, A_name);
            break;
        case DataType::BF16:
            run_flash_sdpa_fwd_cudnn<nntile::bf16_t>(graph, attrs, K_name, Q_name, mask_name, logsumexp_name, V_name, A_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for flash_sdpa_fwd_cudnn operation");
        default:
            throw std::runtime_error("Unsupported data type for flash_sdpa_fwd_cudnn");
    }
}

} // namespace nntile::graph
