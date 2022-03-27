#pragma once

#include <nntile/tile/tile.hh>

#include <Accelerate/Accelerate.h>
//#include <cblas.h>

namespace nntile
{

//! Check if dimensionalities of tensors match gemm
inline void gemm_check_ndim(const TileTraits &A,
        const TileTraits &B,
        const TileTraits &C,
        int ndim=1)
{
    if(ndim <= 0)
    {
        throw std::runtime_error("ndim <= 0");
    }
    if(A.ndim < ndim)
    {
        throw std::runtime_error("A.ndim < ndim");
    }
    if(B.ndim < ndim)
    {
        throw std::runtime_error("B.ndim < ndim");
    }
    if(C.ndim < ndim)
    {
        throw std::runtime_error("C.ndim < ndim");
    }
    if(A.ndim + B.ndim - C.ndim != 2*ndim)
    {
        throw std::runtime_error("A.ndim + B.ndim - C.ndim != 2*ndim");
    }
}

//! Check if shapes of matricized tensors A and B match gemm
inline void gemm_check_A_B(const TileTraits &A,
        const TileTraits &B,
        int ndim=1)
{
    for(int i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-ndim+i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[A.ndim-ndim:A.ndim] != "
                    "B.shape[0:ndim]");
        }
    }
}

//! Check if shapes of matricized tensors A^T and B match gemm
inline void gemm_check_AT_B(const TileTraits &A,
        const TileTraits &B,
        int ndim=1)
{
    for(int i = 0; i < ndim; ++i)
    {
        if(A.shape[i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[0:ndim] != B.shape[0:ndim]");
        }
    }
}

//! Check if shapes of tensors A and B^T match gemm
inline void gemm_check_A_BT(const TileTraits &A,
        const TileTraits &B,
        int ndim=1)
{
    for(int i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-ndim+i] != B.shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.shape[A.ndim-ndim:A.ndim] != "
                    "B.shape[B.ndim-ndim:B.ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B^T match gemm
inline void gemm_check_AT_BT(const TileTraits &A,
        const TileTraits &B,
        int ndim=1)
{
    for(int i = 0; i < ndim; ++i)
    {
        if(A.shape[i] != B.shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.shape[0:ndim] != "
                    "B.shape[B.ndim-ndim:B.ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and op(B) match gemm
inline void gemm_check_opA_opB(const TransOp &transA,
        const TileTraits &A,
        const TransOp &transB,
        const TileTraits &B,
        int ndim=1)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    gemm_check_A_B(A, B, ndim);
                    break;
                case TransOp::Trans:
                    gemm_check_AT_B(A, B, ndim);
                    break;
                default:
                    throw std::runtime_error("Wrong value of transA");
            }
            break;
        case TransOp::Trans:
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    gemm_check_A_BT(A, B, ndim);
                    break;
                case TransOp::Trans:
                    gemm_check_AT_BT(A, B, ndim);
                    break;
                default:
                    throw std::runtime_error("Wrong value of transA");
            }
            break;
        default:
            throw std::runtime_error("Wrong value of transB");
    }
}

//! Check if shapes of tensors A and C match gemm
inline void gemm_check_A_C(const TileTraits &A,
        const TileTraits &C,
        int ndim=1)
{
    for(int i = 0; i < A.ndim-ndim; ++i)
    {
        if(A.shape[i] != C.shape[i])
        {
            throw std::runtime_error("A.shape[0:A.ndim-ndim] != "
                    "C.shape[0:A.ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and C match gemm
inline void gemm_check_AT_C(const TileTraits &A,
        const TileTraits &C,
        int ndim=1)
{
    for(int i = ndim; i < A.ndim-ndim; ++i)
    {
        if(A.shape[i] != C.shape[i-ndim])
        {
            throw std::runtime_error("A.shape[ndim:A.ndim] != "
                    "C.shape[0:A.ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and C match gemm
inline void gemm_check_opA_C(const TransOp &transA,
        const TileTraits &A,
        const TileTraits &C,
        int ndim=1)
{
    switch(transA.value)
    {
        case TransOp::NoTrans:
            gemm_check_A_C(A, C, ndim);
            break;
        case TransOp::Trans:
            gemm_check_AT_C(A, C, ndim);
            break;
        default:
            throw std::runtime_error("Wrong value of transA");
    }
}

//! Check if shapes of tensors B and C match gemm
inline void gemm_check_B_C(const TileTraits &B,
        const TileTraits &C,
        int ndim=1)
{
    for(int i = ndim; i < B.ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim-B.ndim+i])
        {
            throw std::runtime_error("B.shape[ndim:B.ndim] != "
                    "C.shape[C.ndim-B.ndim+ndim:C.ndim]");
        }
    }
}

//! Check if shapes of tensors B^T and C match gemm
inline void gemm_check_BT_C(const TileTraits &B,
        const TileTraits &C,
        int ndim=1)
{
    for(int i = 0; i < B.ndim-ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim-B.ndim+ndim+i])
        {
            throw std::runtime_error("B.shape[0:B.ndim-ndim] != "
                    "C.shape[C.ndim-B.ndim+ndim:C.ndim]");
        }
    }
}

//! Check if shapes of tensors op(B) and C match gemm
inline void gemm_check_opB_C(const TransOp &transB,
        const TileTraits &B,
        const TileTraits &C,
        int ndim=1)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            gemm_check_B_C(B, C, ndim);
            break;
        case TransOp::Trans:
            gemm_check_BT_C(B, C, ndim);
            break;
        default:
            throw std::runtime_error("Wrong value of transB");
    }
}

//! Check if tensors match gemm
void gemm_check(const TransOp &transA,
        const TileTraits &A,
        const TransOp &transB,
        const TileTraits &B,
        const TileTraits &C,
        int ndim=1)
{
    // Check if dimensionalities match
    gemm_check_ndim(A, B, C, ndim);
    // Check if shapes of A and B match
    gemm_check_opA_opB(transA, A, transB, B, ndim);
    // Check if shapes of A and C match
    gemm_check_opA_C(transA, A, C, ndim);
    // Check if shapes of B and C match
    gemm_check_opB_C(transB, B, C, ndim);
}

void cpu_blas_gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
        int M, int N, int K, float alpha, const float *A, int ldA,
        const float *B, int ldB, float beta, float *C, int ldC)
{
    cblas_sgemm(CblasColMajor, transA, transB, M, N, K, alpha, A, ldA, B, ldB,
            beta, C, ldC);
}

void cpu_blas_gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
        int M, int N, int K, double alpha, const double *A, int ldA,
        const double *B, int ldB, double beta, double *C, int ldC)
{
    cblas_dgemm(CblasColMajor, transA, transB, M, N, K, alpha, A, ldA, B, ldB,
            beta, C, ldC);
}

template<typename T>
void gemm_codelet_cpu(void *buffers[], void *cl_args)
{
    struct TransOp transA, transB;
    int m, n, k;
    T alpha, beta;
    starpu_codelet_unpack_args(cl_args, &transA, &transB, &m, &n, &k, &alpha,
            &beta);
    const T *A = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[0]));
    const T *B = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[1]));
    T *C = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[2]));
    CBLAS_TRANSPOSE transA_, transB_;
    int ldA, ldB;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            transA_ = CblasNoTrans;
            ldA = m;
            break;
        case TransOp::Trans:
            transA_ = CblasTrans;
            ldA = k;
            break;
        default:
            throw std::runtime_error("Wrong value of transA");
    }
    switch(transB.value)
    {
        case TransOp::NoTrans:
            transB_ = CblasNoTrans;
            ldB = k;
            break;
        case TransOp::Trans:
            transB_ = CblasTrans;
            ldB = n;
            break;
        default:
            throw std::runtime_error("Wrong value of transA");
    }
    cpu_blas_gemm(transA_, transB_, m, n, k, alpha, A, ldA, B, ldB, beta, C,
            m);
}

template<typename T>
void gemm_async(T alpha,
        const TransOp &transA,
        const TileTraits &A,
        const StarPUSharedHandle &A_handle,
        const TransOp &transB,
        const TileTraits &B,
        const StarPUSharedHandle &B_handle,
        T beta,
        const TileTraits &C,
        const StarPUSharedHandle &C_handle,
        int ndim=1)
{
    static struct starpu_codelet codelet_gemm_w =
    {
        //.where = STARPU_CUDA,
        .cpu_funcs = {gemm_codelet_cpu<T>},
        //.cuda_funcs = {gemm_codelet_gpu_func<T>},
        //.cuda_flags = {STARPU_CUDA_ASYNC},
        .nbuffers = 3,
        .modes = {STARPU_R, STARPU_R, STARPU_W}
    };
    static struct starpu_codelet codelet_gemm_rw =
    {
        //.where = STARPU_CUDA,
        .cpu_funcs = {gemm_codelet_cpu<T>},
        //.cuda_funcs = {gemm_codelet_gpu_func<T>},
        //.cuda_flags = {STARPU_CUDA_ASYNC},
        .nbuffers = 3,
        .modes = {STARPU_R, STARPU_R, STARPU_RW}
    };
    // Check if tensors match gemm
    gemm_check(transA, A, transB, B, C, ndim);
    // Reference tensors as matrices
    int m = C.matrix_shape[A.ndim-ndim-1][0];
    int n = C.matrix_shape[A.ndim-ndim-1][1];
    int k;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            k = A.matrix_shape[A.ndim-ndim-1][1];
            break;
        case TransOp::Trans:
            k = A.matrix_shape[ndim-1][0];
            break;
        default:
            // All parameters were already checked in gemm_check
            break;
    }
    if(beta == 0)
    {
        starpu_task_insert(&codelet_gemm_w,
                STARPU_VALUE, &transA, sizeof(transA),
                STARPU_VALUE, &transB, sizeof(transB),
                STARPU_VALUE, &m, sizeof(m),
                STARPU_VALUE, &n, sizeof(n),
                STARPU_VALUE, &k, sizeof(k),
                STARPU_VALUE, &alpha, sizeof(alpha),
                STARPU_R, starpu_data_handle_t(A_handle),
                STARPU_R, starpu_data_handle_t(B_handle),
                STARPU_VALUE, &beta, sizeof(beta),
                STARPU_W, starpu_data_handle_t(C_handle),
                0);
    }
    else
    {
        starpu_task_insert(&codelet_gemm_rw,
                STARPU_VALUE, &transA, sizeof(transA),
                STARPU_VALUE, &transB, sizeof(transB),
                STARPU_VALUE, &m, sizeof(m),
                STARPU_VALUE, &n, sizeof(n),
                STARPU_VALUE, &k, sizeof(k),
                STARPU_VALUE, &alpha, sizeof(alpha),
                STARPU_R, starpu_data_handle_t(A_handle),
                STARPU_R, starpu_data_handle_t(B_handle),
                STARPU_VALUE, &beta, sizeof(beta),
                STARPU_RW, starpu_data_handle_t(C_handle),
                0);
    }
}

template<typename T>
void gemm_async(T alpha,
        const TransOp &transA,
        const Tile<T> &A,
        const TransOp &transB,
        const Tile<T> &B,
        T beta,
        const Tile<T> &C,
        int ndim=1)
{
    gemm_async(alpha, transA, A, A.handle, B, B.handle, beta, C, C.handle,
            ndim);
}

template<typename T>
void gemm(T alpha,
        const TransOp &transA,
        const TileTraits &A,
        const StarPUSharedHandle &A_handle,
        const TransOp &transB,
        const TileTraits &B,
        const StarPUSharedHandle &B_handle,
        T beta,
        const TileTraits &C,
        const StarPUSharedHandle &C_handle,
        int ndim)
{
    gemm_async(alpha, transA, A, A_handle, transB, B, B_handle, beta, C,
            C_handle, ndim);
    starpu_task_wait_for_all();
}

template<typename T>
void gemm(T alpha,
        const TransOp &transA,
        const Tile<T> &A,
        const TransOp &transB,
        const Tile<T> &B,
        T beta,
        const Tile<T> &C,
        int ndim)
{
    gemm_async(alpha, transA, A, A.handle, transB, B, B.handle, beta, C,
            C.handle, ndim);
    starpu_task_wait_for_all();
}

template<typename T>
void bias_codelet_cpu(void *buffers[], void *cl_args)
{
    int m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    const int mk = m * k;
    T *data = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[0]));
    const T *bias = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[1]));
    int data_offset = 0;
    for(int i2 = 0; i2 < n; ++i2)
    {
        for(int i1 = 0; i1 < k; ++i1)
        {
            int bias_offset = i2 * m;
            for(int i0 = 0; i0 < m; ++i0)
            {
                data[data_offset] += bias[bias_offset];
                ++data_offset;
                ++bias_offset;
            }
        }
    }
}

template<typename T>
void bias_async(const TileTraits &A,
        const StarPUSharedHandle &A_handle,
        const TileTraits &bias,
        const StarPUSharedHandle &bias_handle,
        int batch_dim)
{
    static struct starpu_codelet codelet_bias =
    {
        .cpu_funcs = {bias_codelet_cpu<T>},
        .nbuffers = 2,
        .modes = {STARPU_RW, STARPU_R}
    };
    if(A.ndim-bias.ndim != 1)
    {
        throw std::runtime_error("A.ndim-bias.ndim != 1");
    }
    for(int i = 0; i < batch_dim; ++i)
    {
        if(A.shape[i] != bias.shape[i])
        {
            throw std::runtime_error("A.shape[i] != bias.shape[i]");
        }
    }
    for(int i = batch_dim+1; i < A.ndim; ++i)
    {
        if(A.shape[i] != bias.shape[i-1])
        {
            throw std::runtime_error("A.shape[i] != bias.shape[i-1]");
        }
    }
    int m, n, k;
    if(batch_dim == 0)
    {
        m = 1;
        n = bias.nelems;
        k = A.shape[0];
    }
    else if(batch_dim == A.ndim-1)
    {
        m = bias.nelems;
        n = 1;
        k = A.shape[A.ndim-1];
    }
    else
    {
        m = A.stride[batch_dim];
        n = A.matrix_shape[batch_dim][1];
        k = A.shape[batch_dim];
    }
    starpu_task_insert(&codelet_bias,
            STARPU_VALUE, &m, sizeof(m),
            STARPU_VALUE, &n, sizeof(n),
            STARPU_VALUE, &k, sizeof(k),
            STARPU_RW, starpu_data_handle_t(A_handle),
            STARPU_R, starpu_data_handle_t(bias_handle),
            0);
}

template<typename T>
void bias_async(const Tile<T> &A,
        const Tile<T> &bias,
        int batch_dim)
{
    bias_async<T>(A, A.handle, bias, bias.handle, batch_dim);
}

template<typename T>
void bias(const Tile<T> &A,
        const Tile<T> &bias,
        int batch_dim)
{
    bias_async<T>(A, A.handle, bias, bias.handle, batch_dim);
    starpu_task_wait_for_all();
}

template<typename T>
void bias(const TileTraits &A,
        const StarPUSharedHandle &A_handle,
        const TileTraits &bias,
        const StarPUSharedHandle &bias_handle,
        int batch_dim)
{
    bias_async<T>(A, A_handle, bias, bias_handle, batch_dim);
    starpu_task_wait_for_all();
}

} // namespace nntile

