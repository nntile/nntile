#pragma once

#include <nntile/tensor/tensor.hh>
#include <nntile/tile/gemm.hh>

namespace nntile
{

//! Check if dimensionalities of tensors match gemm
inline void gemm_check_ndim(const TensorTraits &A,
        const TensorTraits &B,
        const TensorTraits &C,
        int ndim=1)
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
    if(C.ndim < ndim_)
    {
        throw std::runtime_error("C.ndim < ndim");
    }
    if(A.ndim + B.ndim != C.ndim + 2*ndim_)
    {
        throw std::runtime_error("A.ndim + B.ndim != C.ndim + 2*ndim");
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
        if(A.basetile_shape[A.ndim-ndim+i] != B.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[A.ndim-ndim:A.ndim] != "
                    "B.basetile_shape[0:ndim]");
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
        if(A.basetile_shape[i] != B.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[0:ndim] != "
                    "B.basetile_shape[0:ndim]");
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
        if(A.basetile_shape[A.ndim-ndim+i] != B.basetile_shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.basetile_shape[A.ndim-ndim:A.ndim] != "
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
        if(A.basetile_shape[i] != B.basetile_shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.basetile_shape[0:ndim] != "
                    "B.basetile_shape[B.ndim-ndim:B.ndim]");
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
        if(A.basetile_shape[i] != C.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[0:A.ndim-ndim] != "
                    "C.basetile_shape[0:A.ndim-ndim]");
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
        if(A.basetile_shape[i] != C.basetile_shape[i-ndim])
        {
            throw std::runtime_error("A.basetile_shape[ndim:A.ndim] != "
                    "C.basetile_shape[0:A.ndim-ndim]");
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
        if(B.basetile_shape[i] != C.basetile_shape[C.ndim-B.ndim+i])
        {
            throw std::runtime_error("B.basetile_shape[ndim:B.ndim] != "
                    "C.basetile_shape[C.ndim-B.ndim+ndim:C.ndim]");
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
        if(B.basetile_shape[i] != C.basetile_shape[C.ndim-B.ndim+ndim+i])
        {
            throw std::runtime_error("B.basetile_shape[0:B.ndim-ndim] != "
                    "C.basetile_shape[C.ndim-B.ndim+ndim:C.ndim]");
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
        const Tensor<T> &A,
        const TransOp &transB,
        const Tensor<T> &B,
        T beta,
        const Tensor<T> &C,
        int ndim=1)
{
    // Check if tensors match gemm
    gemm_check(transA, A, transB, B, C, ndim);
    // Sizes of A, B and C as simple matrices (grids of tiles) for gemm
    size_t m = C.grid.matrix_shape[A.ndim-ndim][0];
    size_t n = C.grid.matrix_shape[A.ndim-ndim][1];
    size_t k;
    std::array<size_t, 2> opA_stride, opB_stride;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            k = A.grid.matrix_shape[A.ndim-ndim][1];
            opA_stride = {1, m};
            break;
        case TransOp::Trans:
            k = A.grid.matrix_shape[ndim][0];
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
    for(size_t j = 0; j < n; ++j)
    {
        for(size_t i = 0; i < m; ++i)
        {
            size_t C_tile_offset = j*m + i;
            const auto &C_tile = C.tiles[C_tile_offset];
            // initialize C(i,j) = a*opA(i,0)*opB(0,j) + b*C(i,j)
            size_t A_tile_offset = opA_stride[0] * i;
            size_t B_tile_offset = opB_stride[1] * j;
            const auto &A_tile = A.tiles[A_tile_offset];
            const auto &B_tile = B.tiles[B_tile_offset];
            gemm_async<T>(alpha, transA, A_tile, transB, B_tile, beta,
                    C_tile, ndim);
            // all other l>0
            for(int l = 1; l < k; ++l)
            {
                // accumulate C(i,j) = a*opA(i,l)*opB(0,l) + C(i,j)
                A_tile_offset += opA_stride[1];
                B_tile_offset += opB_stride[0];
                const auto &A_tile = A.tiles[A_tile_offset];
                const auto &B_tile = B.tiles[B_tile_offset];
                gemm_async<T>(alpha, transA, A_tile, transB, B_tile, beta,
                        C_tile, ndim);
            }
        }
    }
}

template<typename T>
void gemm(T alpha,
        const TransOp &transA,
        const Tensor<T> &A,
        const TransOp &transB,
        const Tensor<T> &B,
        T beta,
        const Tensor<T> &C,
        int ndim=1)
{
    gemm_async(alpha, transA, A, transB, B, beta, C, ndim);
    starpu_task_wait_for_all();
}

} // namespace nntile

