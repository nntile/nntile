/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/gemm.cc
 * GEMM operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/gemm.hh"
#include "nntile/starpu/gemm.hh"
#include "nntile/starpu/config.hh"

namespace nntile::core
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

//! Check batch shapes (leading batch dimensions in C-order layout)
static inline void gemm_check_batch(const TileTraits &A,
        const TileTraits &B, const TileTraits &C, Index batch_ndim)
{
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(A.shape[i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[0:batch_ndim] != "
                    "B.shape[0:batch_ndim]");
        }
        if(A.shape[i] != C.shape[i])
        {
            throw std::runtime_error("A.shape[0:batch_ndim] != "
                    "C.shape[0:batch_ndim]");
        }
    }
}

//! Check if shapes of matricized tensors A and B match gemm
static inline void gemm_check_A_B(const TileTraits &A, const TileTraits &B,
        Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[batch_ndim+i] != B.shape[batch_ndim+i])
        {
            throw std::runtime_error("A.shape[batch_ndim:batch_ndim+ndim] != "
                    "B.shape[batch_ndim:batch_ndim+ndim]");
        }
    }
}

//! Check if shapes of matricized tensors A^T and B match gemm
static inline void gemm_check_AT_B(const TileTraits &A, const TileTraits &B,
        Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-ndim+i] != B.shape[batch_ndim+i])
        {
            throw std::runtime_error("A.shape[A.ndim-ndim:A.ndim] != "
                    "B.shape[batch_ndim:batch_ndim+ndim]");
        }
    }
}

//! Check if shapes of tensors A and B^T match gemm
static inline void gemm_check_A_BT(const TileTraits &A, const TileTraits &B,
        Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[batch_ndim+i] != B.shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.shape[batch_ndim:batch_ndim+ndim] != "
                    "B.shape[B.ndim-ndim:B.ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B^T match gemm
static inline void gemm_check_AT_BT(const TileTraits &A, const TileTraits &B,
        Index ndim, Index batch_ndim)
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
                    gemm_check_AT_B(A, B, ndim, batch_ndim);
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
    const Index num_m = A.ndim - batch_ndim - ndim;
    const Index c_m_begin = C.ndim - num_m;
    for(Index i = 0; i < num_m; ++i)
    {
        if(A.shape[batch_ndim+ndim+i] != C.shape[c_m_begin+i])
        {
            throw std::runtime_error("A.shape[batch_ndim+ndim:"
                    "A.ndim] != C.shape[C.ndim-num_m:C.ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and C match gemm
static inline void gemm_check_AT_C(const TileTraits &A, const TileTraits &C,
        Index ndim, Index batch_ndim)
{
    const Index num_m = A.ndim - batch_ndim - ndim;
    const Index c_m_begin = C.ndim - num_m;
    for(Index i = 0; i < num_m; ++i)
    {
        if(A.shape[batch_ndim+i] != C.shape[c_m_begin+i])
        {
            throw std::runtime_error("A.shape[batch_ndim:A.ndim-ndim] != "
                    "C.shape[C.ndim-num_m:C.ndim]");
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
    const Index num_n = B.ndim - batch_ndim - ndim;
    for(Index i = 0; i < num_n; ++i)
    {
        if(B.shape[batch_ndim+ndim+i] != C.shape[batch_ndim+i])
        {
            throw std::runtime_error("B.shape[batch_ndim+ndim:B.ndim] != "
                    "C.shape[batch_ndim:batch_ndim+num_n]");
        }
    }
}

//! Check if shapes of tensors B^T and C match gemm
static inline void gemm_check_BT_C(const TileTraits &B, const TileTraits &C,
        Index ndim, Index batch_ndim)
{
    const Index num_n = B.ndim - batch_ndim - ndim;
    for(Index i = 0; i < num_n; ++i)
    {
        if(B.shape[batch_ndim+i] != C.shape[batch_ndim+i])
        {
            throw std::runtime_error("B.shape[batch_ndim:B.ndim-ndim] != "
                    "C.shape[batch_ndim:batch_ndim+num_n]");
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
void gemm_async(int starpu_worker_hint, Scalar alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, Scalar beta, const Tile<T> &C,
        Index ndim, Index batch_ndim, int redux)
{
    // Check inputs (throw exception in case of an error)
    gemm_check(transA, A, transB, B, C, ndim, batch_ndim);
    // Reference tensors as matrices (C-order: [batch..., K..., M/N layout])
    const Index split = batch_ndim + ndim;
    Index batch = 1;
    for(Index i = 0; i < batch_ndim; ++i)
    {
        batch *= C.shape[i];
    }
    const Index m = C.matrix_shape[split][1];
    const Index n = C.matrix_shape[split][0] / batch;
    Index k = 1;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            for(Index i = 0; i < ndim; ++i)
            {
                k *= A.shape[batch_ndim+i];
            }
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            for(Index i = 0; i < ndim; ++i)
            {
                k *= A.shape[A.ndim-ndim+i];
            }
            break;
    }
    // Insert task
    int mpi_rank = starpu_mpi_world_rank();
    int c_rank = C.mpi_get_rank();
    A.mpi_transfer(c_rank, mpi_rank);
    B.mpi_transfer(c_rank, mpi_rank);
    if(mpi_rank == c_rank)
    {
        starpu::gemm.submit<std::tuple<T>>(starpu_worker_hint, 
            transA, transB, m, n, k, batch, alpha, A, B, beta, C, 0);  // redux ignored for now
    }
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
void gemm(int starpu_worker_hint, Scalar alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, Scalar beta, const Tile<T> &C,
        Index ndim, Index batch_ndim, int redux)
{
    gemm_async<T>(starpu_worker_hint, alpha, transA, A, transB, B, beta, C, ndim, batch_ndim,
            redux);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void gemm_async<fp32_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<fp32_t> &A,
        const TransOp &transB, const Tile<fp32_t> &B, Scalar beta,
        const Tile<fp32_t> &C, Index ndim, Index batch_ndim, int redux);

template
void gemm_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<fp32_fast_tf32_t> &A,
        const TransOp &transB, const Tile<fp32_fast_tf32_t> &B, Scalar beta,
        const Tile<fp32_fast_tf32_t> &C, Index ndim, Index batch_ndim,
        int redux);

template
void gemm_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<fp32_fast_fp16_t> &A,
        const TransOp &transB, const Tile<fp32_fast_fp16_t> &B, Scalar beta,
        const Tile<fp32_fast_fp16_t> &C, Index ndim, Index batch_ndim,
        int redux);

template
void gemm_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<fp32_fast_bf16_t> &A,
        const TransOp &transB, const Tile<fp32_fast_bf16_t> &B, Scalar beta,
        const Tile<fp32_fast_bf16_t> &C, Index ndim, Index batch_ndim,
        int redux);

template
void gemm_async<fp64_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<fp64_t> &A,
        const TransOp &transB, const Tile<fp64_t> &B, Scalar beta,
        const Tile<fp64_t> &C, Index ndim, Index batch_ndim, int redux);

template
void gemm_async<bf16_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<bf16_t> &A,
        const TransOp &transB, const Tile<bf16_t> &B, Scalar beta,
        const Tile<bf16_t> &C, Index ndim, Index batch_ndim, int redux);

template
void gemm_async<fp16_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<fp16_t> &A,
        const TransOp &transB, const Tile<fp16_t> &B, Scalar beta,
        const Tile<fp16_t> &C, Index ndim, Index batch_ndim, int redux);

// Explicit instantiation
template
void gemm<fp32_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<fp32_t> &A,
        const TransOp &transB, const Tile<fp32_t> &B, Scalar beta,
        const Tile<fp32_t> &C, Index ndim, Index batch_ndim, int redux);

template
void gemm<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<fp32_fast_tf32_t> &A,
        const TransOp &transB, const Tile<fp32_fast_tf32_t> &B, Scalar beta,
        const Tile<fp32_fast_tf32_t> &C, Index ndim, Index batch_ndim,
        int redux);

template
void gemm<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<fp32_fast_fp16_t> &A,
        const TransOp &transB, const Tile<fp32_fast_fp16_t> &B, Scalar beta,
        const Tile<fp32_fast_fp16_t> &C, Index ndim, Index batch_ndim,
        int redux);

template
void gemm<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<fp32_fast_bf16_t> &A,
        const TransOp &transB, const Tile<fp32_fast_bf16_t> &B, Scalar beta,
        const Tile<fp32_fast_bf16_t> &C, Index ndim, Index batch_ndim,
        int redux);

template
void gemm<fp64_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<fp64_t> &A,
        const TransOp &transB, const Tile<fp64_t> &B, Scalar beta,
        const Tile<fp64_t> &C, Index ndim, Index batch_ndim, int redux);

template
void gemm<bf16_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<bf16_t> &A,
        const TransOp &transB, const Tile<bf16_t> &B, Scalar beta,
        const Tile<bf16_t> &C, Index ndim, Index batch_ndim, int redux);

template
void gemm<fp16_t>(int starpu_worker_hint, Scalar alpha, const TransOp &transA,
        const Tile<fp16_t> &A,
        const TransOp &transB, const Tile<fp16_t> &B, Scalar beta,
        const Tile<fp16_t> &C, Index ndim, Index batch_ndim, int redux);

} // namespace nntile::core
