/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/gemm.cc
 * GEMM operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/gemm.hh"
#include "nntile/starpu/gemm.hh"

namespace nntile::tile
{

//! Check if dimensionalities of tensors match gemm
static inline void gemm_check_ndim(const TileTraits &A, const TileTraits &B,
        const TileTraits &C, Index ndim, Index batch_ndim)
{
    // Check if ndim is negative since it will be converted to Index
    if(ndim <= 0)
    {
        throw std::runtime_error("ndim <= 0");
    }
    if(batch_ndim < 0)
    {
        throw std::runtime_error("batch_ndim < 0");
    }
    if(A.ndim < batch_ndim+ndim)
    {
        throw std::runtime_error("A.ndim < batch_ndim+ndim");
    }
    if(B.ndim < batch_ndim+ndim)
    {
        throw std::runtime_error("B.ndim < batch_ndim+ndim");
    }
    if(A.ndim + B.ndim != C.ndim + 2*ndim + batch_ndim)
    {
        throw std::runtime_error("A.ndim + B.ndim != C.ndim + 2*ndim + "
                "batch_ndim");
    }
}

//! Check batch shapes
static inline void gemm_check_batch(const TileTraits &A,
        const TileTraits &B, const TileTraits &C, Index batch_ndim)
{
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(A.shape[A.ndim-i-1] != B.shape[B.ndim-i-1])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim:A.ndim] != "
                    "B.shape[B.ndim-batch_ndim:B.ndim]");
        }
        if(A.shape[A.ndim-i-1] != C.shape[C.ndim-i-1])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim:A.ndim] != "
                    "C.shape[C.ndim-batch_ndim:C.ndim]");
        }
    }
}

//! Check if shapes of matricized tensors A and B match gemm
static inline void gemm_check_A_B(const TileTraits &A, const TileTraits &B,
        Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-batch_ndim-ndim+i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim-ndim:"
                    "A.ndim-batch_ndim] != B.shape[0:ndim]");
        }
    }
}

//! Check if shapes of matricized tensors A^T and B match gemm
static inline void gemm_check_AT_B(const TileTraits &A, const TileTraits &B,
        Index ndim)
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
        Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-batch_ndim-ndim+i]
                != B.shape[B.ndim-batch_ndim-ndim+i])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim-ndim:"
                    "A.ndim-batch_ndim] != B.shape[B.ndim-batch_ndim-ndim:"
                    "B.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B^T match gemm
static inline void gemm_check_AT_BT(const TileTraits &A, const TileTraits &B,
        Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[i] != B.shape[B.ndim-batch_ndim-ndim+i])
        {
            throw std::runtime_error("A.shape[0:ndim] != "
                    "B.shape[B.ndim-batch_ndim-ndim:B.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and op(B) match gemm
static inline void gemm_check_opA_opB(const TransOp &transA,
        const TileTraits &A, const TransOp &transB, const TileTraits &B,
        Index ndim, Index batch_ndim)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    gemm_check_A_B(A, B, ndim, batch_ndim);
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
                    gemm_check_A_BT(A, B, ndim, batch_ndim);
                    break;
                case TransOp::Trans:
                    gemm_check_AT_BT(A, B, ndim, batch_ndim);
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
        Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < A.ndim-batch_ndim-ndim; ++i)
    {
        if(A.shape[i] != C.shape[i])
        {
            throw std::runtime_error("A.shape[0:A.ndim-batch_ndim-ndim] != "
                    "C.shape[0:A.ndim-batch_ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and C match gemm
static inline void gemm_check_AT_C(const TileTraits &A, const TileTraits &C,
        Index ndim, Index batch_ndim)
{
    for(Index i = ndim; i < A.ndim-batch_ndim; ++i)
    {
        if(A.shape[i] != C.shape[i-ndim])
        {
            throw std::runtime_error("A.shape[ndim:A.ndim-batch_ndim] != "
                    "C.shape[0:A.ndim-batch_ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and C match gemm
static inline void gemm_check_opA_C(const TransOp &transA, const TileTraits &A,
        const TileTraits &C, Index ndim, Index batch_ndim)
{
    switch(transA.value)
    {
        case TransOp::NoTrans:
            gemm_check_A_C(A, C, ndim, batch_ndim);
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            gemm_check_AT_C(A, C, ndim, batch_ndim);
    }
}

//! Check if shapes of tensors B and C match gemm
static inline void gemm_check_B_C(const TileTraits &B, const TileTraits &C,
        Index ndim, Index batch_ndim)
{
    for(Index i = ndim; i < B.ndim-batch_ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim-B.ndim+i])
        {
            throw std::runtime_error("B.shape[ndim:B.ndim-batch_ndim] != "
                    "C.shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors B^T and C match gemm
static inline void gemm_check_BT_C(const TileTraits &B, const TileTraits &C,
        Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < B.ndim-batch_ndim-ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim-B.ndim+ndim+i])
        {
            throw std::runtime_error("B.shape[0:B.ndim-batch_ndim-ndim] != "
                    "C.shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors op(B) and C match gemm
static inline void gemm_check_opB_C(const TransOp &transB, const TileTraits &B,
        const TileTraits &C, Index ndim, Index batch_ndim)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            gemm_check_B_C(B, C, ndim, batch_ndim);
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            gemm_check_BT_C(B, C, ndim, batch_ndim);
    }
}

//! Check if tensors match gemm
void gemm_check(const TransOp &transA, const TileTraits &A,
        const TransOp &transB, const TileTraits &B, const TileTraits &C,
        Index ndim, Index batch_ndim)
{
    // Check if dimensionalities match
    gemm_check_ndim(A, B, C, ndim, batch_ndim);
    // Check if batch shapes match
    gemm_check_batch(A, B, C, batch_ndim);
    // Check if shapes of A and B match gemm
    gemm_check_opA_opB(transA, A, transB, B, ndim, batch_ndim);
    // Check if shapes of A and C match gemm
    gemm_check_opA_C(transA, A, C, ndim, batch_ndim);
    // Check if shapes of B and C match gemm
    gemm_check_opB_C(transB, B, C, ndim, batch_ndim);
}

//! Asynchronous tile-wise gemm operation
/*! @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tile A
 * @param[in] A: Input tile A
 * @param[in] transB: Transposition flag for the tile B
 * @param[in] B: Input tile B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tile C
 * @param[in] ndim: Number of dimensions used in gemm contraction
 * @param[in] batch_ndim: Number of last dimensions used for batching of gemms
 * */
template<typename T>
void gemm_async(Scalar alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, Scalar beta, const Tile<T> &C,
        Index ndim, Index batch_ndim)
{
    // Check inputs (throw exception in case of an error)
    gemm_check(transA, A, transB, B, C, ndim, batch_ndim);
    // Reference tensors as matrices
    Index m = C.matrix_shape[A.ndim-batch_ndim-ndim][0];
    Index batch = C.matrix_shape[C.ndim-batch_ndim][1];
    Index n = C.matrix_shape[A.ndim-batch_ndim-ndim][1] / batch;
    Index k;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            k = A.matrix_shape[A.ndim-batch_ndim-ndim][1] / batch;
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            k = A.matrix_shape[ndim][0];
            break;
    }
    // Insert task
    starpu::gemm::submit<T>(transA, transB, m, n, k, batch, alpha, A,
            B, beta, C);
}

//! Blocking version of tile-wise gemm operation
/*! @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tile A
 * @param[in] A: Input tile A
 * @param[in] transB: Transposition flag for the tile B
 * @param[in] B: Input tile B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tile C
 * @param[in] ndim: Number of dimensions used in gemm contraction
 * @param[in] batch_ndim: Number of last dimensions used for batching of gemms
 * */
template<typename T>
void gemm(Scalar alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, Scalar beta, const Tile<T> &C,
        Index ndim, Index batch_ndim)
{
    gemm_async<T>(alpha, transA, A, transB, B, beta, C, ndim, batch_ndim);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void gemm_async<fp32_t>(Scalar alpha, const TransOp &transA,
        const Tile<fp32_t> &A,
        const TransOp &transB, const Tile<fp32_t> &B, Scalar beta,
        const Tile<fp32_t> &C, Index ndim, Index batch_ndim);

template
void gemm_async<fp32_fast_tf32_t>(Scalar alpha, const TransOp &transA,
        const Tile<fp32_fast_tf32_t> &A,
        const TransOp &transB, const Tile<fp32_fast_tf32_t> &B, Scalar beta,
        const Tile<fp32_fast_tf32_t> &C, Index ndim, Index batch_ndim);

template
void gemm_async<fp64_t>(Scalar alpha, const TransOp &transA,
        const Tile<fp64_t> &A,
        const TransOp &transB, const Tile<fp64_t> &B, Scalar beta,
        const Tile<fp64_t> &C, Index ndim, Index batch_ndim);

//template
//void gemm_async<fp16_t>(Scalar alpha, const TransOp &transA,
//        const Tile<fp16_t> &A,
//        const TransOp &transB, const Tile<fp16_t> &B, Scalar beta,
//        const Tile<fp16_t> &C, Index ndim, Index batch_ndim);

// Explicit instantiation
template
void gemm<fp32_t>(Scalar alpha, const TransOp &transA,
        const Tile<fp32_t> &A,
        const TransOp &transB, const Tile<fp32_t> &B, Scalar beta,
        const Tile<fp32_t> &C, Index ndim, Index batch_ndim);

template
void gemm<fp32_fast_tf32_t>(Scalar alpha, const TransOp &transA,
        const Tile<fp32_fast_tf32_t> &A,
        const TransOp &transB, const Tile<fp32_fast_tf32_t> &B, Scalar beta,
        const Tile<fp32_fast_tf32_t> &C, Index ndim, Index batch_ndim);

template
void gemm<fp64_t>(Scalar alpha, const TransOp &transA,
        const Tile<fp64_t> &A,
        const TransOp &transB, const Tile<fp64_t> &B, Scalar beta,
        const Tile<fp64_t> &C, Index ndim, Index batch_ndim);

template
void gemm<bf16_t>(Scalar alpha, const TransOp &transA,
        const Tile<bf16_t> &A,
        const TransOp &transB, const Tile<bf16_t> &B, Scalar beta,
        const Tile<bf16_t> &C, Index ndim, Index batch_ndim);

//template
//void gemm<fp16_t>(Scalar alpha, const TransOp &transA,
//        const Tile<fp16_t> &A,
//        const TransOp &transB, const Tile<fp16_t> &B, Scalar beta,
//        const Tile<fp16_t> &C, Index ndim, Index batch_ndim);

} // namespace nntile::tile
