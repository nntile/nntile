/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/gemm.cc
 * GEMM operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/gemm.hh"
#include "nntile/starpu/gemm.hh"

namespace nntile::tensor
{

//! Check if dimensionalities of tensors match gemm
static inline void gemm_check_ndim(const TensorTraits &A,
        const TensorTraits &B, const TensorTraits &C, Index ndim,
        Index batch_ndim)
{
    // Check if ndim is negative since it will be converted to Index
    if(ndim < 0)
    {
        throw std::runtime_error("ndim < 0");
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
static inline void gemm_check_batch(const TensorTraits &A,
        const TensorTraits &B, const TensorTraits &C, Index batch_ndim)
{
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(A.shape[A.ndim-i-1] != B.shape[B.ndim-i-1])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim:A.ndim] != "
                    "B.shape[B.ndim-batch_ndim:B.ndim]");
        }
        if(A.basetile_shape[A.ndim-i-1] != B.basetile_shape[B.ndim-i-1])
        {
            throw std::runtime_error("A.basetile_shape[A.ndim-batch_ndim:"
                    "A.ndim] != B.basetile_shape[B.ndim-batch_ndim:B.ndim]");
        }
        if(A.shape[A.ndim-i-1] != C.shape[C.ndim-i-1])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim:A.ndim] != "
                    "C.shape[C.ndim-batch_ndim:C.ndim]");
        }
        if(A.basetile_shape[A.ndim-i-1] != C.basetile_shape[C.ndim-i-1])
        {
            throw std::runtime_error("A.basetile_shape[A.ndim-batch_ndim:"
                    "A.ndim] != C.basetile_shape[C.ndim-batch_ndim:C.ndim]");
        }
    }
}

//! Check if shapes of tensors A and B match gemm
static inline void gemm_check_A_B(const TensorTraits &A,
        const TensorTraits &B, Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-batch_ndim-ndim+i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim-ndim:"
                    "A.ndim-batch_ndim] != B.shape[0:ndim]");
        }
        if(A.basetile_shape[A.ndim-batch_ndim-ndim+i] != B.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[A.ndim-batch_ndim-ndim:"
                    "A.ndim-batch_ndim] != B.basetile_shape[0:ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B match gemm
static inline void gemm_check_AT_B(const TensorTraits &A,
        const TensorTraits &B, Index ndim)
{
    for(Index i = 0; i < ndim; ++i)
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
static inline void gemm_check_A_BT(const TensorTraits &A,
        const TensorTraits &B, Index ndim, Index batch_ndim)
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
        if(A.basetile_shape[A.ndim-batch_ndim-ndim+i]
                != B.basetile_shape[B.ndim-batch_ndim-ndim+i])
        {
            throw std::runtime_error("A.basetile_shape[A.ndim-batch_ndim-ndim:"
                    "A.ndim-batch_ndim] != B.shape[B.ndim-batch_ndim-ndim:"
                    "B.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B^T match gemm
static inline void gemm_check_AT_BT(const TensorTraits &A,
        const TensorTraits &B, Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[i] != B.shape[B.ndim-batch_ndim-ndim+i])
        {
            throw std::runtime_error("A.shape[0:ndim] != "
                    "B.shape[B.ndim-batch_ndim-ndim:B.ndim-batch_ndim]");
        }
        if(A.basetile_shape[i] != B.basetile_shape[B.ndim-batch_ndim-ndim+i])
        {
            throw std::runtime_error("A.basetile_shape[0:ndim] != "
                    "B.basetile_shape[B.ndim-batch_ndim-ndim:"
                    "B.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and op(B) match gemm
static inline void gemm_check_opA_opB(const TransOp &transA,
        const TensorTraits &A, const TransOp &transB, const TensorTraits &B,
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
static inline void gemm_check_A_C(const TensorTraits &A,
        const TensorTraits &C, Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < A.ndim-batch_ndim-ndim; ++i)
    {
        if(A.shape[i] != C.shape[i])
        {
            throw std::runtime_error("A.shape[0:A.ndim-batch_ndim-ndim] != "
                    "C.shape[0:A.ndim-batch_ndim-ndim]");
        }
        if(A.basetile_shape[i] != C.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[0:"
                    "A.ndim-batch_ndim-ndim] != "
                    "C.basetile_shape[0:A.ndim-batch_ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and C match gemm
static inline void gemm_check_AT_C(const TensorTraits &A,
        const TensorTraits &C, Index ndim, Index batch_ndim)
{
    for(Index i = ndim; i < A.ndim-batch_ndim; ++i)
    {
        if(A.shape[i] != C.shape[i-ndim])
        {
            throw std::runtime_error("A.shape[ndim:A.ndim-batch_ndim] != "
                    "C.shape[0:A.ndim-batch_ndim-ndim]");
        }
        if(A.basetile_shape[i] != C.basetile_shape[i-ndim])
        {
            throw std::runtime_error("A.basetile_shape[ndim:"
                    "A.ndim-batch_ndim] != "
                    "C.basetile_shape[0:A.ndim-batch_ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and C match gemm
static inline void gemm_check_opA_C(const TransOp &transA,
        const TensorTraits &A, const TensorTraits &C, Index ndim,
        Index batch_ndim)
{
    switch(transA.value)
    {
        case TransOp::NoTrans:
            gemm_check_A_C(A, C, ndim, batch_ndim);
            break;
        case TransOp::Trans:
            gemm_check_AT_C(A, C, ndim, batch_ndim);
            break;
        // This parameter was already checked in gemm_check_opA_opB
    }
}

//! Check if shapes of tensors B and C match gemm
static inline void gemm_check_B_C(const TensorTraits &B,
        const TensorTraits &C, Index ndim, Index batch_ndim)
{
    for(Index i = ndim; i < B.ndim-batch_ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim-B.ndim+i])
        {
            throw std::runtime_error("B.shape[ndim:B.ndim-batch_ndim] != "
                    "C.shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
        if(B.basetile_shape[i] != C.basetile_shape[C.ndim-B.ndim+i])
        {
            throw std::runtime_error("B.basetile_shape[ndim:"
                    "B.ndim-batch_ndim] != "
                    "C.basetile_shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors B^T and C match gemm
static inline void gemm_check_BT_C(const TensorTraits &B,
        const TensorTraits &C, Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < B.ndim-batch_ndim-ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim-B.ndim+ndim+i])
        {
            throw std::runtime_error("B.shape[0:B.ndim-batch_ndim-ndim] != "
                    "C.shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
        if(B.basetile_shape[i] != C.basetile_shape[C.ndim-B.ndim+ndim+i])
        {
            throw std::runtime_error("B.basetile_shape[0:"
                    "B.ndim-batch_ndim-ndim] != "
                    "C.basetile_shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors op(B) and C match gemm
static inline void gemm_check_opB_C(const TransOp &transB,
        const TensorTraits &B, const TensorTraits &C, Index ndim,
        Index batch_ndim)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            gemm_check_B_C(B, C, ndim, batch_ndim);
            break;
        case TransOp::Trans:
            gemm_check_BT_C(B, C, ndim, batch_ndim);
            break;
        // This parameter was already checked in gemm_check_opA_opB
    }
}

//! Check if tensors match gemm
void gemm_check(const TransOp &transA, const TensorTraits &A,
        const TransOp &transB, const TensorTraits &B, const TensorTraits &C,
        Index ndim, Index batch_ndim)
{
    // Check if dimensionalities match
    gemm_check_ndim(A, B, C, ndim, batch_ndim);
    // Check if batch shapes match
    gemm_check_batch(A, B, C, batch_ndim);
    // Check if shapes of A and B match
    gemm_check_opA_opB(transA, A, transB, B, ndim, batch_ndim);
    // Check if shapes of A and C match
    gemm_check_opA_C(transA, A, C, ndim, batch_ndim);
    // Check if shapes of B and C match
    gemm_check_opB_C(transB, B, C, ndim, batch_ndim);
}

//! Asynchronous version of tensor-wise gemm operation
/*! Matrix multiplication for tensors, which are virtually reshaped
 *
 * @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tensor A
 * @param[in] A: Input tensor A
 * @param[in] transB: Transposition flag for the tensor B
 * @param[in] B: Input tensor B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tensor C
 * @param[in] ndim: Number of dimensions used in gemm contraction
 * @param[in] batch_ndim: Number of last dimensions used for batching of gemms
 * @param[in] redux: Whether or not to use STARPU_REDUX
 * */
template<typename T>
void gemm_async(Scalar alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, Scalar beta,
        const Tensor<T> &C, Index ndim, Index batch_ndim, int redux)
{
    // Check inputs (throw exception in case of an error)
    gemm_check(transA, A, transB, B, C, ndim, batch_ndim);
    // Sizes of A, B and C as simple matrices (grids of tiles) for gemm
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    constexpr Scalar one = 1;
    Index m = C.grid.matrix_shape[A.ndim-batch_ndim-ndim][0];
    Index batch = C.grid.matrix_shape[C.ndim-batch_ndim][1];
    Index n = C.grid.matrix_shape[A.ndim-batch_ndim-ndim][1] / batch;
    Index k;
    std::array<Index, 2> opA_stride, opB_stride;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            k = A.grid.matrix_shape[A.ndim-batch_ndim-ndim][1] / batch;
            opA_stride = {1, m};
            break;
        case TransOp::Trans:
            k = A.grid.matrix_shape[ndim][0];
            opA_stride = {k, 1};
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
    }
    // All per-tile starpu gemm calls shall appear here
    for(Index b = 0; b < batch; ++b)
    {
        for(Index j = 0; j < n; ++j)
        {
            for(Index i = 0; i < m; ++i)
            {
                Index C_tile_offset = (b*n+j)*m + i;
                auto C_tile_handle = C.get_tile_handle(C_tile_offset);
                auto C_tile_traits = C.get_tile_traits(C_tile_offset);
                int C_tile_rank = C_tile_handle.mpi_get_rank();
                Index tile_m = C_tile_traits.matrix_shape[
                    A.ndim-batch_ndim-ndim][0];
                Index tile_batch = C_tile_traits.matrix_shape[
                    C.ndim-batch_ndim][1];
                Index tile_n = C_tile_traits.matrix_shape[
                    A.ndim-batch_ndim-ndim][1] / tile_batch;
                // initialize C(i,j,b) = a*opA(i,0,b)*opB(0,j,b) + b*C(i,j,b)
                Index A_tile_offset = opA_stride[0]*i + b*m*k;
                Index B_tile_offset = opB_stride[1]*j + b*n*k;
                auto A_first_tile_handle = A.get_tile_handle(A_tile_offset);
                auto B_first_tile_handle = B.get_tile_handle(B_tile_offset);
                int A_first_tile_rank = A_first_tile_handle.mpi_get_rank();
                int B_first_tile_rank = B_first_tile_handle.mpi_get_rank();
                // Transfer first tile A on node with tile C
                A_first_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                // Transfer first tile B on node with tile C
                B_first_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                // Execute on node with tile C
                if(mpi_rank == C_tile_rank)
                {
                    Index tile_k;
                    auto A_first_tile_traits = A.get_tile_traits(
                            A_tile_offset);
                    switch(transA.value)
                    {
                        case TransOp::NoTrans:
                            tile_k = A_first_tile_traits.matrix_shape[
                                A.ndim-batch_ndim-ndim][1] / tile_batch;
                            break;
                            // This parameter was already checked
                            //case TransOp::Trans:
                        default:
                            tile_k = A_first_tile_traits.matrix_shape[ndim][0];
                            break;
                    }
                    starpu::gemm::submit<T>(transA, transB, tile_m,
                            tile_n,
                            tile_k, tile_batch, alpha, A_first_tile_handle,
                            B_first_tile_handle, beta, C_tile_handle, redux);
                }
                // all other l>0
                for(Index l = 1; l < k; ++l)
                {
                    // accumulate C(i,j,b) = a*opA(i,l,b)*opB(l,j,b) + C(i,j,b)
                    A_tile_offset += opA_stride[1];
                    B_tile_offset += opB_stride[0];
                    auto A_tile_handle = A.get_tile_handle(A_tile_offset);
                    auto B_tile_handle = B.get_tile_handle(B_tile_offset);
                    int A_tile_rank = A_tile_handle.mpi_get_rank();
                    int B_tile_rank = B_tile_handle.mpi_get_rank();
                    // Transfer tile A on node with tile C
                    A_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                    // Transfer tile B on node with tile C
                    B_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                    // Execute on node with tile C
                    if(mpi_rank == C_tile_rank)
                    {
                        Index tile_k;
                        auto A_tile_traits = A.get_tile_traits(A_tile_offset);
                        switch(transA.value)
                        {
                            case TransOp::NoTrans:
                                tile_k = A_tile_traits.matrix_shape[
                                    A.ndim-batch_ndim-ndim][1] / tile_batch;
                                break;
                                // This parameter was already checked
                                //case TransOp::Trans:
                            default:
                                tile_k = A_tile_traits.matrix_shape[ndim][0];
                                break;
                        }
                        starpu::gemm::submit<T>(transA, transB, tile_m,
                                tile_n,
                                tile_k, tile_batch, alpha, A_tile_handle,
                                B_tile_handle, one, C_tile_handle, redux);
                    }
                }
                // Flush cache for the output tile on every node
                C_tile_handle.mpi_flush();
            }
        }
    }
}

//! Blocking version of tensor-wise gemm operation
/*! Matrix multiplication for tensors, which are virtually reshaped
 *
 * @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tensor A
 * @param[in] A: Input tensor A
 * @param[in] transB: Transposition flag for the tensor B
 * @param[in] B: Input tensor B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tensor C
 * @param[in] ndim: Number of dimensions used in gemm contraction
 * @param[in] batch_ndim: Number of last dimensions used for batching of gemms
 * */
template<typename T>
void gemm(Scalar alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, Scalar beta,
        const Tensor<T> &C, Index ndim, Index batch_ndim, int redux)
{
    gemm_async<T>(alpha, transA, A, transB, B, beta, C, ndim,
            batch_ndim, redux);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void gemm_async<fp32_t>(Scalar alpha, const TransOp &transA,
        const Tensor<fp32_t> &A,
        const TransOp &transB, const Tensor<fp32_t> &B, Scalar beta,
        const Tensor<fp32_t> &C, Index ndim, Index batch_ndim, int redux);

template
void gemm_async<fp32_fast_tf32_t>(Scalar alpha, const TransOp &transA,
        const Tensor<fp32_fast_tf32_t> &A,
        const TransOp &transB, const Tensor<fp32_fast_tf32_t> &B, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &C, Index ndim, Index batch_ndim, int redux);

template
void gemm_async<fp64_t>(Scalar alpha, const TransOp &transA,
        const Tensor<fp64_t> &A,
        const TransOp &transB, const Tensor<fp64_t> &B, Scalar beta,
        const Tensor<fp64_t> &C, Index ndim, Index batch_ndim, int redux);

//template
//void gemm_async<fp16_t>(Scalar alpha, const TransOp &transA,
//        const Tensor<fp16_t> &A,
//        const TransOp &transB, const Tensor<fp16_t> &B, Scalar beta,
//        const Tensor<fp16_t> &C, Index ndim, Index batch_ndim, int redux);

// Explicit instantiation
template
void gemm<fp32_t>(Scalar alpha, const TransOp &transA,
        const Tensor<fp32_t> &A,
        const TransOp &transB, const Tensor<fp32_t> &B, Scalar beta,
        const Tensor<fp32_t> &C, Index ndim, Index batch_ndim, int redux);

template
void gemm<fp32_fast_tf32_t>(Scalar alpha, const TransOp &transA,
        const Tensor<fp32_fast_tf32_t> &A,
        const TransOp &transB, const Tensor<fp32_fast_tf32_t> &B, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &C, Index ndim, Index batch_ndim, int redux);

template
void gemm<fp64_t>(Scalar alpha, const TransOp &transA,
        const Tensor<fp64_t> &A,
        const TransOp &transB, const Tensor<fp64_t> &B, Scalar beta,
        const Tensor<fp64_t> &C, Index ndim, Index batch_ndim, int redux);

template
void gemm<bf16_t>(Scalar alpha, const TransOp &transA,
        const Tensor<bf16_t> &A,
        const TransOp &transB, const Tensor<bf16_t> &B, Scalar beta,
        const Tensor<bf16_t> &C, Index ndim, Index batch_ndim, int redux);

//template
//void gemm<fp16_t>(Scalar alpha, const TransOp &transA,
//        const Tensor<fp16_t> &A,
//        const TransOp &transB, const Tensor<fp16_t> &B, Scalar beta,
//        const Tensor<fp16_t> &C, Index ndim, Index batch_ndim, int redux);

} // namespace nntile::tensor
