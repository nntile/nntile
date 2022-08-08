/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/gemm.cc
 * GEMM operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-08
 * */

#include "nntile/tile/gemm.hh"
#include "nntile/starpu/gemm.hh"

namespace nntile
{

//! Check if dimensionalities of tensors match gemm
static inline void gemm_check_ndim(const TileTraits &A, const TileTraits &B,
        const TileTraits &C, Index ndim=1)
{
    // Check if ndim is negative since it will be converted to Index
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
    if(A.ndim + B.ndim != C.ndim + 2*ndim)
    {
        throw std::runtime_error("A.ndim + B.ndim != C.ndim + 2*ndim");
    }
}

//! Check if shapes of matricized tensors A and B match gemm
static inline void gemm_check_A_B(const TileTraits &A, const TileTraits &B,
        Index ndim=1)
{
    for(Index i = 0; i < ndim; ++i)
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
        Index ndim=1)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[0:ndim] != B.shape[0:ndim]");
        }
    }
}

//! Check if shapes of tensors A and B^T match gemm
static inline void gemm_check_A_BT(const TileTraits &A, const TileTraits &B,
        Index ndim=1)
{
    for(Index i = 0; i < ndim; ++i)
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
        Index ndim=1)
{
    for(Index i = 0; i < ndim; ++i)
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
        Index ndim=1)
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
        Index ndim=1)
{
    for(Index i = 0; i < A.ndim-ndim; ++i)
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
        Index ndim=1)
{
    for(Index i = ndim; i < A.ndim; ++i)
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
        const TileTraits &C, Index ndim=1)
{
    switch(transA.value)
    {
        case TransOp::NoTrans:
            gemm_check_A_C(A, C, ndim);
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            gemm_check_AT_C(A, C, ndim);
    }
}

//! Check if shapes of tensors B and C match gemm
static inline void gemm_check_B_C(const TileTraits &B, const TileTraits &C,
        Index ndim=1)
{
    for(Index i = ndim; i < B.ndim; ++i)
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
        Index ndim=1)
{
    for(Index i = 0; i < B.ndim-ndim; ++i)
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
        const TileTraits &C, Index ndim=1)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            gemm_check_B_C(B, C, ndim);
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            gemm_check_BT_C(B, C, ndim);
    }
}

//! Check if tensors match gemm
void gemm_check(const TransOp &transA, const TileTraits &A,
        const TransOp &transB, const TileTraits &B, const TileTraits &C,
        Index ndim)
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
void gemm_work(T alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, T beta, const Tile<T> &C,
        Index ndim)
{
    // Reference tensors as matrices
    Index m = C.matrix_shape[A.ndim-ndim][0];
    Index n = C.matrix_shape[A.ndim-ndim][1];
    Index k;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            k = A.matrix_shape[A.ndim-ndim][1];
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            k = A.matrix_shape[ndim][0];
            break;
    }
    nntile::starpu::gemm<T>(transA, transB, m, n, k, alpha, A, B, beta, C);
}

// Explicit instantiation of templates
template
void gemm_work(fp32_t alpha, const TransOp &transA, const Tile<fp32_t> &A,
        const TransOp &transB, const Tile<fp32_t> &B, fp32_t beta,
        const Tile<fp32_t> &C, Index ndim);

template
void gemm_work(fp64_t alpha, const TransOp &transA, const Tile<fp64_t> &A,
        const TransOp &transB, const Tile<fp64_t> &B, fp64_t beta,
        const Tile<fp64_t> &C, Index ndim);

} // namespace nntile

