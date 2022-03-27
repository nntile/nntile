#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

//! Check if dimensionalities of tensors match gemm
inline void gemm_check_ndim(const TensorTraits &A,
        const TensorTraits &B,
        const TensorTraits &C,
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

//! Check if shapes of tensors A and B match gemm
inline void gemm_check_A_B(const TensorTraits &A,
        const TensorTraits &B,
        int ndim=1)
{
    for(int i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-ndim+i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[A.ndim-ndim:A.ndim] != "
                    "B.shape[0:ndim]");
        }
        if(A.tile_shape[A.ndim-ndim+i] != B.tile_shape[i])
        {
            throw std::runtime_error("A.tile_shape[A.ndim-ndim:A.ndim] != "
                    "B.tile_shape[0:ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B match gemm
inline void gemm_check_AT_B(const TensorTraits &A,
        const TensorTraits &B,
        int ndim=1)
{
    for(int i = 0; i < ndim; ++i)
    {
        if(A.shape[i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[0:ndim] != B.shape[0:ndim]");
        }
        if(A.tile_shape[i] != B.tile_shape[i])
        {
            throw std::runtime_error("A.tile_shape[0:ndim] != "
                    "B.tile_shape[0:ndim]");
        }
    }
}

//! Check if shapes of tensors A and B^T match gemm
inline void gemm_check_A_BT(const TensorTraits &A,
        const TensorTraits &B,
        int ndim=1)
{
    for(int i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-ndim+i] != B.shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.shape[A.ndim-ndim:A.ndim] != "
                    "B.shape[B.ndim-ndim:B.ndim]");
        }
        if(A.tile_shape[A.ndim-ndim+i] != B.tile_shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.tile_shape[A.ndim-ndim:A.ndim] != "
                    "B.shape[B.ndim-ndim:B.ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B^T match gemm
inline void gemm_check_AT_BT(const TensorTraits &A,
        const TensorTraits &B,
        int ndim=1)
{
    for(int i = 0; i < ndim; ++i)
    {
        if(A.shape[i] != B.shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.shape[0:ndim] != "
                    "B.shape[B.ndim-ndim:B.ndim]");
        }
        if(A.tile_shape[i] != B.tile_shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.tile_shape[0:ndim] != "
                    "B.tile_shape[B.ndim-ndim:B.ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and op(B) match gemm
inline void gemm_check_opA_opB(const TransOp &transA,
        const TensorTraits &A,
        const TransOp &transB,
        const TensorTraits &B,
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
inline void gemm_check_A_C(const TensorTraits &A,
        const TensorTraits &C,
        int ndim=1)
{
    for(int i = 0; i < A.ndim-ndim; ++i)
    {
        if(A.shape[i] != C.shape[i])
        {
            throw std::runtime_error("A.shape[0:A.ndim-ndim] != "
                    "C.shape[0:A.ndim-ndim]");
        }
        if(A.tile_shape[i] != C.tile_shape[i])
        {
            throw std::runtime_error("A.tile_shape[0:A.ndim-ndim] != "
                    "C.tile_shape[0:A.ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and C match gemm
inline void gemm_check_AT_C(const TensorTraits &A,
        const TensorTraits &C,
        int ndim=1)
{
    for(int i = ndim; i < A.ndim-ndim; ++i)
    {
        if(A.shape[i] != C.shape[i-ndim])
        {
            throw std::runtime_error("A.shape[ndim:A.ndim] != "
                    "C.shape[0:A.ndim-ndim]");
        }
        if(A.tile_shape[i] != C.tile_shape[i-ndim])
        {
            throw std::runtime_error("A.tile_shape[ndim:A.ndim] != "
                    "C.tile_shape[0:A.ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and C match gemm
inline void gemm_check_opA_C(const TransOp &transA,
        const TensorTraits &A,
        const TensorTraits &C,
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
inline void gemm_check_B_C(const TensorTraits &B,
        const TensorTraits &C,
        int ndim=1)
{
    for(int i = ndim; i < B.ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim-B.ndim+i])
        {
            throw std::runtime_error("B.shape[ndim:B.ndim] != "
                    "C.shape[C.ndim-B.ndim+ndim:C.ndim]");
        }
        if(B.tile_shape[i] != C.tile_shape[C.ndim-B.ndim+i])
        {
            throw std::runtime_error("B.tile_shape[ndim:B.ndim] != "
                    "C.tile_shape[C.ndim-B.ndim+ndim:C.ndim]");
        }
    }
}

//! Check if shapes of tensors B^T and C match gemm
inline void gemm_check_BT_C(const TensorTraits &B,
        const TensorTraits &C,
        int ndim=1)
{
    for(int i = 0; i < B.ndim-ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim-B.ndim+ndim+i])
        {
            throw std::runtime_error("B.shape[0:B.ndim-ndim] != "
                    "C.shape[C.ndim-B.ndim+ndim:C.ndim]");
        }
        if(B.tile_shape[i] != C.tile_shape[C.ndim-B.ndim+ndim+i])
        {
            throw std::runtime_error("B.tile_shape[0:B.ndim-ndim] != "
                    "C.tile_shape[C.ndim-B.ndim+ndim:C.ndim]");
        }
    }
}

//! Check if shapes of tensors op(B) and C match gemm
inline void gemm_check_opB_C(const TransOp &transB,
        const TensorTraits &B,
        const TensorTraits &C,
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
        const TensorTraits &A,
        const TransOp &transB,
        const TensorTraits &B,
        const TensorTraits &C,
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

template<typename T>
void gemm_async(T alpha,
        const TransOp &transA,
        const TensorTraits &A,
        const std::vector<StarPUSharedHandle> &A_tiles_handle,
        const TransOp &transB,
        const TensorTraits &B,
        const std::vector<StarPUSharedHandle> &B_tiles_handle,
        T beta,
        const TensorTraits &C,
        const std::vector<StarPUSharedHandle> &C_tiles_handle,
        int ndim=1)
{
    // Check if tensors match gemm
    gemm_check(transA, A, transB, B, C, ndim);
    // Sizes of A, B and C as simple matrices (grids of tiles) for gemm
    int m = C.grid_matrix_shape[A.ndim-ndim-1][0];
    int n = C.grid_matrix_shape[A.ndim-ndim-1][1];
    int k;
    std::array<int, 2> opA_stride, opB_stride;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            k = A.grid_matrix_shape[A.ndim-ndim-1][1];
            opA_stride = {1, m};
            break;
        case TransOp::Trans:
            k = A.grid_matrix_shape[ndim-1][0];
            opA_stride = {k, 1};
            break;
        default:
            // All parameters were already checked in gemm_check
            break;
    }
    switch(transB.value)
    {
        case TransOp::NoTrans:
            opB_stride = {1, k};
            break;
        case TransOp::Trans:
            opB_stride = {n, 1};
            break;
        default:
            // All parameters were already checked in gemm_check
            break;
    }
    // All per-tile gemm_async calls shall appear here
    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < m; ++i)
        {
            int C_tile_offset = j*m + i;
            const auto &C_tile_traits = C.tiles_traits[C_tile_offset];
            const auto &C_tile_handle = C_tiles_handle[C_tile_offset];
            // initialize C(i,j) = a*opA(i,0)*opB(0,j) + b*C(i,j)
            int A_tile_offset = opA_stride[0] * i;
            int B_tile_offset = opB_stride[1] * j;
            const auto &A_tile_traits = A.tiles_traits[A_tile_offset];
            const auto &A_tile_handle = A_tiles_handle[A_tile_offset];
            const auto &B_tile_traits = B.tiles_traits[B_tile_offset];
            const auto &B_tile_handle = B_tiles_handle[B_tile_offset];
            std::cout << "i=" << i << " j=" << j << "\n"
                << "m=" << m << " n=" << n << " k=" << k << "\n"
                << "offsets:"
                << " " << A_tile_offset
                << " " << B_tile_offset
                << " " << C_tile_offset << "\n"
                << "A\n" << A_tile_traits
                << "B\n" << B_tile_traits
                << "C\n" << C_tile_traits << "\n";
            gemm_async<T>(alpha, transA, A_tile_traits, A_tile_handle,
                    transB, B_tile_traits, B_tile_handle,
                    beta, C_tile_traits, C_tile_handle, ndim);
//            // all other l>0
//            for(int l = 1; l < k; ++l)
//            {
//                // accumulate C(i,j) = a*opA(i,l)*opB(0,l) + C(i,j)
//                A_tile_offset = opA_stride[0]*i + opA_stride[1]*l;
//                B_tile_offset = opB_stride[0]*l + opB_stride[1]*j;
//                auto &A_tile_traits = A.tiles_traits[A_tile_offset];
//                auto &A_tile_handle = A_tiles_handle[A_tile_offset];
//                auto &B_tile_traits = B.tiles_traits[B_tile_offset];
//                auto &B_tile_handle = B_tiles_handle[B_tile_offset];
//            }
        }
    }
}

template<typename T>
void gemm_async(T alpha,
        const TransOp &transA,
        const Tensor<T> &A,
        const TransOp &transB,
        const Tensor<T> &B,
        T beta,
        const Tensor<T> &C,
        int ndim=1)
{
    gemm_async(alpha, transA, A, A.tiles_handle, B, B.tiles_handle, beta, C,
            C.tiles_handle, ndim);
}

template<typename T>
void gemm(T alpha,
        const TransOp &transA,
        const TensorTraits &A,
        const std::vector<StarPUSharedHandle> &A_tiles_handle,
        const TransOp &transB,
        const TensorTraits &B,
        const std::vector<StarPUSharedHandle> &B_tiles_handle,
        T beta,
        const TensorTraits &C,
        const std::vector<StarPUSharedHandle> &C_tiles_handle,
        int ndim)
{
    gemm_async(alpha, transA, A, A_tiles_handle, transB, B, B_tiles_handle,
            beta, C, C_tiles_handle, ndim);
    starpu_task_wait_for_all();
}

template<typename T>
void gemm(T alpha,
        const TransOp &transA,
        const Tensor<T> &A,
        const TransOp &transB,
        const Tensor<T> &B,
        T beta,
        const Tensor<T> &C,
        int ndim)
{
    gemm_async(alpha, transA, A, A.tiles_handle, transB, B, B.tiles_handle,
            beta, C, C.tiles_handle, ndim);
    starpu_task_wait_for_all();
}

} // namespace nntile

