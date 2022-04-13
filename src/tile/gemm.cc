#include "nntile/tile/gemm.hh"
#include <type_traits>

#include "nntile/defs.h"
//#define NNTILE_USE_APPLE_ACCELERATE

#if defined(NNTILE_USE_APPLE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#   define CBLAS_INT int
#elif defined(NNTILE_USE_INTEL_MKL)
#   include <mkl.h>
#else
#   include <cblas.h>
#endif

// Define type CBLAS_INT to use cblas properly
#ifndef CBLAS_INT
#   if defined(f77_int)
#      define CBLAS_INT f77_int
#   elif defined(CBLAS_INDEX)
#       define CBLAS_INT CBLAS_INDEX
#   endif
#endif

namespace nntile
{

//! Check if dimensionalities of tensors match gemm
static inline void gemm_check_ndim(const TileTraits &A, const TileTraits &B,
        const TileTraits &C, int ndim=1)
{
    // Check if ndim is negative since it will be converted to size_t
    if(ndim <= 0)
    {
        throw std::runtime_error("ndim <= 0");
    }
    size_t ndim_ = ndim;
    if(A.ndim < ndim_)
    {
        throw std::runtime_error("A.ndim < ndim");
    }
    if(B.ndim < ndim_)
    {
        throw std::runtime_error("B.ndim < ndim");
    }
    if(A.ndim + B.ndim != C.ndim + 2*ndim_)
    {
        throw std::runtime_error("A.ndim + B.ndim != C.ndim + 2*ndim");
    }
}

//! Check if shapes of matricized tensors A and B match gemm
static inline void gemm_check_A_B(const TileTraits &A, const TileTraits &B,
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
static inline void gemm_check_AT_B(const TileTraits &A, const TileTraits &B,
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
static inline void gemm_check_A_BT(const TileTraits &A, const TileTraits &B,
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
static inline void gemm_check_AT_BT(const TileTraits &A, const TileTraits &B,
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
static inline void gemm_check_opA_opB(const TransOp &transA,
        const TileTraits &A, const TransOp &transB, const TileTraits &B,
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
static inline void gemm_check_A_C(const TileTraits &A, const TileTraits &C,
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
static inline void gemm_check_AT_C(const TileTraits &A, const TileTraits &C,
        int ndim=1)
{
    for(int i = ndim; i < A.ndim; ++i)
    {
        if(A.shape[i] != C.shape[i-ndim])
        {
            throw std::runtime_error("A.shape[ndim:A.ndim] != "
                    "C.shape[0:A.ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and C match gemm
static inline void gemm_check_opA_C(const TransOp &transA, const TileTraits &A,
        const TileTraits &C, int ndim=1)
{
    switch(transA.value)
    {
        case TransOp::NoTrans:
            gemm_check_A_C(A, C, ndim);
            break;
        case TransOp::Trans:
            gemm_check_AT_C(A, C, ndim);
            break;
        // This parameter was already checked in gemm_check_opA_opB
    }
}

//! Check if shapes of tensors B and C match gemm
static inline void gemm_check_B_C(const TileTraits &B, const TileTraits &C,
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
static inline void gemm_check_BT_C(const TileTraits &B, const TileTraits &C,
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
static inline void gemm_check_opB_C(const TransOp &transB, const TileTraits &B,
        const TileTraits &C, int ndim=1)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            gemm_check_B_C(B, C, ndim);
            break;
        case TransOp::Trans:
            gemm_check_BT_C(B, C, ndim);
            break;
        // This parameter was already checked in gemm_check_opA_opB
    }
}

//! Check if tensors match gemm
static inline void gemm_check(const TransOp &transA, const TileTraits &A,
        const TransOp &transB, const TileTraits &B, const TileTraits &C,
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

static inline void cblas_gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
        CBLAS_INT M, CBLAS_INT N, CBLAS_INT K, float alpha, const float *A,
        CBLAS_INT ldA, const float *B, CBLAS_INT ldB, float beta, float *C,
        CBLAS_INT ldC)
{
    cblas_sgemm(CblasColMajor, transA, transB, M, N, K, alpha, A, ldA, B, ldB,
            beta, C, ldC);
}

static inline void cblas_gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
        CBLAS_INT M, CBLAS_INT N, CBLAS_INT K, double alpha, const double *A,
        CBLAS_INT ldA, const double *B, CBLAS_INT ldB, double beta, double *C,
        CBLAS_INT ldC)
{
    cblas_dgemm(CblasColMajor, transA, transB, M, N, K, alpha, A, ldA, B, ldB,
            beta, C, ldC);
}

template<typename T>
static void gemm_codelet_cpu(void *buffers[], void *cl_args)
{
    TransOp::Value transA_value, transB_value;
    size_t m, n, k;
    T alpha, beta;
    starpu_codelet_unpack_args(cl_args, &transA_value, &transB_value, &m, &n,
            &k, &alpha, &beta);
    const T *A = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    const T *B = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    T *C = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[2]));
    CBLAS_TRANSPOSE transA_, transB_;
    CBLAS_INT M=m, N=n, K=k, ldA, ldB;
    switch(transA_value)
    {
        case TransOp::NoTrans:
            transA_ = CblasNoTrans;
            ldA = M;
            break;
        case TransOp::Trans:
            transA_ = CblasTrans;
            ldA = K;
            break;
        // This parameter was already checked in gemm_check_opA_opB
    }
    switch(transB_value)
    {
        case TransOp::NoTrans:
            transB_ = CblasNoTrans;
            ldB = K;
            break;
        case TransOp::Trans:
            transB_ = CblasTrans;
            ldB = N;
            break;
        // This parameter was already checked in gemm_check_opA_opB
    }
    cblas_gemm(transA_, transB_, M, N, K, alpha, A, ldA, B, ldB, beta, C,
            M);
}

template<typename T>
void gemm_async(T alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, T beta, const Tile<T> &C,
        int ndim)
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
    constexpr auto commute_mode = static_cast<enum starpu_data_access_mode>(
            STARPU_RW | STARPU_COMMUTE);
    static struct starpu_codelet codelet_gemm_rw_commute =
    {
        //.where = STARPU_CUDA,
        .cpu_funcs = {gemm_codelet_cpu<T>},
        //.cuda_funcs = {gemm_codelet_gpu_func<T>},
        //.cuda_flags = {STARPU_CUDA_ASYNC},
        .nbuffers = 3,
        .modes = {STARPU_R, STARPU_R, commute_mode}
    };
    // Check if tensors match gemm
    gemm_check(transA, A, transB, B, C, ndim);
    // Reference tensors as matrices
    size_t m = C.matrix_shape[A.ndim-ndim][0];
    size_t n = C.matrix_shape[A.ndim-ndim][1];
    size_t k;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            k = A.matrix_shape[A.ndim-ndim][1];
            break;
        case TransOp::Trans:
            k = A.matrix_shape[ndim][0];
            break;
        // This parameter was already checked in gemm_check_opA_opB
    }
    // Check that matrix sizes fit proper types for underlying CBLAS
    // Ignore code coverage on the following lines
    // LCOV_EXCL_START
#if defined(NNTILE_USE_CBLAS)
    if(static_cast<CBLAS_INT>(m) != m)
    {
        throw std::runtime_error("GEMM size M does not fit CBLAS_INT");
    }
    if(static_cast<CBLAS_INT>(n) != n)
    {
        throw std::runtime_error("GEMM size N does not fit CBLAS_INT");
    }
    if(static_cast<CBLAS_INT>(k) != k)
    {
        throw std::runtime_error("GEMM size K does not fit CBLAS_INT");
    }
#endif
    // Check that matrix sizes fit proper types for underlying CUBLAS
#if defined(NNTILE_USE_CUBLAS)
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
#endif
    // LCOV_EXCL_STOP
    constexpr T zero = 0, one = 1;
    if(beta == zero)
    {
        starpu_task_insert(&codelet_gemm_w,
                STARPU_VALUE, &transA.value, sizeof(transA.value),
                STARPU_VALUE, &transB.value, sizeof(transB.value),
                STARPU_VALUE, &m, sizeof(m),
                STARPU_VALUE, &n, sizeof(n),
                STARPU_VALUE, &k, sizeof(k),
                STARPU_VALUE, &alpha, sizeof(alpha),
                STARPU_R, static_cast<starpu_data_handle_t>(A),
                STARPU_R, static_cast<starpu_data_handle_t>(B),
                STARPU_VALUE, &beta, sizeof(beta),
                STARPU_W, static_cast<starpu_data_handle_t>(C),
                0);
    }
    else if(beta == one)
    {
        starpu_task_insert(&codelet_gemm_rw_commute,
                STARPU_VALUE, &transA.value, sizeof(transA.value),
                STARPU_VALUE, &transB.value, sizeof(transB.value),
                STARPU_VALUE, &m, sizeof(m),
                STARPU_VALUE, &n, sizeof(n),
                STARPU_VALUE, &k, sizeof(k),
                STARPU_VALUE, &alpha, sizeof(alpha),
                STARPU_R, static_cast<starpu_data_handle_t>(A),
                STARPU_R, static_cast<starpu_data_handle_t>(B),
                STARPU_VALUE, &beta, sizeof(beta),
                STARPU_RW | STARPU_COMMUTE,
                static_cast<starpu_data_handle_t>(C),
                0);
    }
    else
    {
        starpu_task_insert(&codelet_gemm_rw,
                STARPU_VALUE, &transA.value, sizeof(transA.value),
                STARPU_VALUE, &transB.value, sizeof(transB.value),
                STARPU_VALUE, &m, sizeof(m),
                STARPU_VALUE, &n, sizeof(n),
                STARPU_VALUE, &k, sizeof(k),
                STARPU_VALUE, &alpha, sizeof(alpha),
                STARPU_R, static_cast<starpu_data_handle_t>(A),
                STARPU_R, static_cast<starpu_data_handle_t>(B),
                STARPU_VALUE, &beta, sizeof(beta),
                STARPU_RW, static_cast<starpu_data_handle_t>(C),
                0);
    }
}

template
void gemm_async<float>(float alpha, const TransOp &transA,
        const Tile<float> &A, const TransOp &transB, const Tile<float> &B,
        float beta, const Tile<float> &C, int ndim=1);

template
void gemm_async<double>(double alpha, const TransOp &transA,
        const Tile<double> &A, const TransOp &transB, const Tile<double> &B,
        double beta, const Tile<double> &C, int ndim=1);

} // namespace nntile

