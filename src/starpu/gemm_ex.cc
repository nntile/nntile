/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gemm_ex.cc
 * GEMM extended operations for StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-02
 * */

#include "nntile/starpu/gemm_ex.hh"

#ifdef NNTILE_USE_CUDA
#   include <cublas_v2.h>
#   include <starpu_cublas_v2.h>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace starpu
{
namespace gemm_ex
{

#ifdef NNTILE_USE_CUDA
// Overloaded call to cuBLAS GEMM
static inline
void cublas(cublasHandle_t handle, cublasOperation_t transA,
        cublasOperation_t transB, int M, int N, int K, fp32_t alpha,
        const fp32_t *A, int ldA, const fp32_t *B, int ldB, fp32_t beta,
        fp32_t *C, int ldC)
    noexcept
{
    cublasGemmEx(handle, transA, transB, M, N, K, &alpha, A, CUDA_R_32F, ldA,
            B, CUDA_R_32F, ldB, &beta, C, CUDA_R_32F, ldC,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

//! GEMM for contiguous matrices without padding through StarPU buffers
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    // Launch kernel
    const T *A = interfaces[0]->get_ptr<T>();
    const T *B = interfaces[1]->get_ptr<T>();
    T *C = interfaces[2]->get_ptr<T>();
    // It is OK to convert values as it was checked during task submission
    int M=args->m, N=args->n, K=args->k, ldA, ldB, ldC=M;
    cublasOperation_t transA_, transB_;
    // Convert other values to CBLAS types
    switch(args->transA.value)
    {
        case TransOp::NoTrans:
            transA_ = CUBLAS_OP_N;
            ldA = M;
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            transA_ = CUBLAS_OP_T;
            ldA = K;
    }
    switch(args->transB.value)
    {
        case TransOp::NoTrans:
            transB_ = CUBLAS_OP_N;
            ldB = K;
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            transB_ = CUBLAS_OP_T;
            ldB = N;
    }
    // Get cuBLAS handle and CUDA stream
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasSetStream(handle, stream);
    // alpha and beta parameters of GEMM operation are on CPU host
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    // Call corresponding cuBLAS routine
    Index A_offset = args->m * args->k, B_offset = args->n * args->k,
            C_offset = args->m * args->n;
    for(Index i = 0; i < args->batch; ++i)
    {
        cublas(handle, transA_, transB_, M, N, K, args->alpha, A, ldA, B, ldB,
                args->beta, C, M);
        A += A_offset;
        B += B_offset;
        C += C_offset;
    }
}
#endif //NNTILE_USE_CUDA

//! Footprint for GEMM tasks that depends only on M, N, K and alpha
template<typename T>
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(task->cl_arg);
    // In case alpha is zero, entire gemm is unnecessary so it is better to
    // give it a different footprint since gemm time will be totally different
    uint32_t hash = args->alpha == T{0} ? -1 : 0;
    // Apply hash over parameters M, N and K. This way if we swap values of M,
    // N and K total size of buffers will remain the same, but the footprint
    // will be different
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    hash = starpu_hash_crc32c_be_n(&args->batch, sizeof(args->batch), hash);
    return hash;
}

Codelet codelet_NN_fp32, codelet_NT_fp32, codelet_TN_fp32, codelet_TT_fp32;

void init()
{
    codelet_NN_fp32.init("nntile_gemm_ex_NN_fp32_fast_fp16",
            footprint<fp32_t>,
            {},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_NT_fp32.init("nntile_gemm_ex_NT_fp32_fast_fp16",
            footprint<fp32_t>,
            {},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_TN_fp32.init("nntile_gemm_ex_TN_fp32_fast_fp16",
            footprint<fp32_t>,
            {},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_TT_fp32.init("nntile_gemm_ex_TT_fp32_fast_fp16",
            footprint<fp32_t>,
            {},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
}

void restrict_where(uint32_t where)
{
    codelet_NN_fp32.restrict_where(where);
    codelet_NT_fp32.restrict_where(where);
    codelet_TN_fp32.restrict_where(where);
    codelet_TT_fp32.restrict_where(where);
}

void restore_where()
{
    codelet_NN_fp32.restore_where();
    codelet_NT_fp32.restore_where();
    codelet_TN_fp32.restore_where();
    codelet_TT_fp32.restore_where();
}

template<typename T>
void submit(const TransOp &transA, const TransOp &transB, Index m, Index n,
        Index k, Index batch, T alpha, Handle A, Handle B, T beta, Handle C)
{
    // Check that matrix sizes fit proper types for underlying CUBLAS
#ifdef NNTILE_USE_CUDA
    if(static_cast<int>(m) != m)
    {
        throw std::runtime_error("GEMM size M does not fit int");
    }
    if(static_cast<int>(n) != n)
    {
        throw std::runtime_error("GEMM size N does not fit int");
    }
    if(static_cast<int>(k) != k)
    {
        throw std::runtime_error("GEMM size K does not fit int");
    }
#endif // NNTILE_USE_CUDA
    constexpr T zero = 0, one = 1;
    enum starpu_data_access_mode C_mode;
    if(beta == zero)
    {
        C_mode = STARPU_W;
    }
    else if(beta == one)
    {
        C_mode = Config::STARPU_RW_COMMUTE;
    }
    else
    {
        C_mode = STARPU_RW;
    }
    // Codelet arguments
    auto args = new args_t<T>
    {
        .transA = transA,
        .transB = transB,
        .m = m,
        .n = n,
        .k = k,
        .batch = batch,
        .alpha = alpha,
        .beta = beta
    };
    fp64_t nflops = 2 * m * n * k;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(transA, transB),
            STARPU_R, static_cast<starpu_data_handle_t>(A),
            STARPU_R, static_cast<starpu_data_handle_t>(B),
            C_mode, static_cast<starpu_data_handle_t>(C),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in gemm_ex task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(const TransOp &transA, const TransOp &transB, Index m,
        Index n, Index k, Index batch, fp32_t alpha, Handle A, Handle B,
        fp32_t beta, Handle C);

} // namespace gemm_ex
} // namespace starpu
} // namespace nntile

