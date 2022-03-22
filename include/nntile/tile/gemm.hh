#pragma once

#include <nntile/tile/tile.hh>

#include <Accelerate/Accelerate.h>

namespace nntile
{

template<typename T, typename Ta, typename Tb, typename TA, typename TB>
void gemm(Ta alpha, const TA &transA, const ContiguousTile<T> &A,
        const TB &transB, const ContiguousTile<T> &B, Tb beta,
        ContiguousTile<T> &C, int ndim, const struct Debug::Debug &)
{
    gemm_check(transA, A, transB, B, C, ndim);
    gemm(alpha, transA, A, transB, B, beta, C, ndim);
}

template<typename T, typename Ta, typename Tb, typename TA, typename TB>
void gemm(Ta alpha, const TA &transA, const ContiguousTile<T> &A,
        const TB &transB, const ContiguousTile<T> &B, Tb beta,
        ContiguousTile<T> &C, int ndim, const struct Debug::NoDebug &)
{
    gemm(alpha, transA, A, transB, B, beta, C, ndim);
}

template<typename T>
void gemm(T alpha, const struct TransOp &transA,
        const ContiguousTile<T> &A, const struct TransOp &transB,
        const ContiguousTile<T> &B, T beta, ContiguousTile<T> &C, int ndim)
{
    gemm_async(alpha, transA, A, transB, B, beta, C, ndim);
    starpu_task_wait_for_all();
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
void gemm_async(T alpha, const struct TransOp &transA,
        const ContiguousTile<T> &A, const struct TransOp &transB,
        const ContiguousTile<T> &B, T beta, ContiguousTile<T> &C,
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
    // Reference tensors as matrices
    int m = C.matrix_shape[A.ndim-ndim-1][0];
    int n = C.matrix_shape[A.ndim-ndim-1][1];
    int k;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            if(m != A.matrix_shape[A.ndim-ndim-1][0])
            {
                throw std::runtime_error("Shapes of A and C are incorrect");
            }
            k = A.matrix_shape[A.ndim-ndim-1][1];
            break;
        case TransOp::Trans:
            if(m != A.matrix_shape[ndim-1][1])
            {
                throw std::runtime_error("Shapes of A and C are incorrect");
            }
            k = A.matrix_shape[ndim-1][0];
            break;
        default:
            throw std::runtime_error("Wrong value of transA");
    }
    switch(transB.value)
    {
        case TransOp::NoTrans:
            if(k != B.matrix_shape[ndim-1][0])
            {
                throw std::runtime_error("Shapes of A and B are incorrect");
            }
            if(n != B.matrix_shape[ndim-1][1])
            {
                throw std::runtime_error("Shapes of B and C are incorrect");
            }
            break;
        case TransOp::Trans:
            if(k != B.matrix_shape[B.ndim-ndim-1][1])
            {
                throw std::runtime_error("Shapes of A and B are incorrect");
            }
            if(n != B.matrix_shape[B.ndim-ndim-1][0])
            {
                throw std::runtime_error("Shapes of B and C are incorrect");
            }
            break;
        default:
            throw std::runtime_error("Wrong value of transB");
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
                STARPU_R, A.handle,
                STARPU_R, B.handle,
                STARPU_VALUE, &beta, sizeof(beta),
                STARPU_W, C.handle,
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
                STARPU_R, A.handle,
                STARPU_R, B.handle,
                STARPU_VALUE, &beta, sizeof(beta),
                STARPU_RW, C.handle,
                0);
    }
}

template<typename T>
void bias(ContiguousTile<T> &A, const ContiguousTile<T> &bias, int batch_dim)
{
    bias_async(A, bias, batch_dim);
    starpu_task_wait_for_all();
}

template<typename T>
void bias_codelet_cpu(void *buffers[], void *cl_args)
{
    int m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    int mk = m * k;
    T *data = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[0]));
    T *bias = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[1]));
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
void bias_async(ContiguousTile<T> &A, const ContiguousTile<T> &bias,
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
    std::cout << "M=" << m << " N=" << n << " K=" << k << "\n";
    starpu_task_insert(&codelet_bias,
            STARPU_VALUE, &m, sizeof(m),
            STARPU_VALUE, &n, sizeof(n),
            STARPU_VALUE, &k, sizeof(k),
            STARPU_RW, A.handle,
            STARPU_R, bias.handle,
            0);
}

} // namespace nntile

