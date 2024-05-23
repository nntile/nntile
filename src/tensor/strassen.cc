/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 * *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/strassen.cc
 * Strassen operation for Tensor<T>
 *
 * @version 1.0.0
 * */

#include "nntile/starpu/add.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/gemm.hh"
#include "nntile/tensor/strassen.hh"

using nntile::starpu::Handle;

namespace nntile
{
namespace tensor
{

//! Check if dimensionalities of tensors match strassen
static inline void strassen_check_ndim(const TensorTraits &A,
                                       const TensorTraits &B,
                                       const TensorTraits &C, Index ndim,
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
    if(A.ndim < batch_ndim + ndim)
    {
        throw std::runtime_error("A.ndim < batch_ndim+ndim");
    }
    if(B.ndim < batch_ndim + ndim)
    {
        throw std::runtime_error("B.ndim < batch_ndim+ndim");
    }
    if(A.ndim + B.ndim != C.ndim + 2 * ndim + batch_ndim)
    {
        throw std::runtime_error("A.ndim + B.ndim != C.ndim + 2*ndim + "
                                 "batch_ndim");
    }
}

//! Check batch shapes
static inline void strassen_check_batch(const TensorTraits &A,
                                        const TensorTraits &B,
                                        const TensorTraits &C, Index batch_ndim)
{
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(A.shape[A.ndim - i - 1] != B.shape[B.ndim - i - 1])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim:A.ndim] != "
                                     "B.shape[B.ndim-batch_ndim:B.ndim]");
        }
        if(A.basetile_shape[A.ndim - i - 1] != B.basetile_shape[B.ndim - i - 1])
        {
            throw std::runtime_error(
                "A.basetile_shape[A.ndim-batch_ndim:"
                "A.ndim] != B.basetile_shape[B.ndim-batch_ndim:B.ndim]");
        }
        if(A.shape[A.ndim - i - 1] != C.shape[C.ndim - i - 1])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim:A.ndim] != "
                                     "C.shape[C.ndim-batch_ndim:C.ndim]");
        }
        if(A.basetile_shape[A.ndim - i - 1] != C.basetile_shape[C.ndim - i - 1])
        {
            throw std::runtime_error(
                "A.basetile_shape[A.ndim-batch_ndim:"
                "A.ndim] != C.basetile_shape[C.ndim-batch_ndim:C.ndim]");
        }
    }
}

//! Check if shapes of tensors A and B match strassen
static inline void strassen_check_A_B(const TensorTraits &A,
                                      const TensorTraits &B, Index ndim,
                                      Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim - batch_ndim - ndim + i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim-ndim:"
                                     "A.ndim-batch_ndim] != B.shape[0:ndim]");
        }
        if(A.basetile_shape[A.ndim - batch_ndim - ndim + i] !=
           B.basetile_shape[i])
        {
            throw std::runtime_error(
                "A.basetile_shape[A.ndim-batch_ndim-ndim:"
                "A.ndim-batch_ndim] != B.basetile_shape[0:ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B match strassen
static inline void strassen_check_AT_B(const TensorTraits &A,
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

//! Check if shapes of tensors A and B^T match strassen
static inline void strassen_check_A_BT(const TensorTraits &A,
                                       const TensorTraits &B, Index ndim,
                                       Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim - batch_ndim - ndim + i] !=
           B.shape[B.ndim - batch_ndim - ndim + i])
        {
            throw std::runtime_error(
                "A.shape[A.ndim-batch_ndim-ndim:"
                "A.ndim-batch_ndim] != B.shape[B.ndim-batch_ndim-ndim:"
                "B.ndim-batch_ndim]");
        }
        if(A.basetile_shape[A.ndim - batch_ndim - ndim + i] !=
           B.basetile_shape[B.ndim - batch_ndim - ndim + i])
        {
            throw std::runtime_error(
                "A.basetile_shape[A.ndim-batch_ndim-ndim:"
                "A.ndim-batch_ndim] != B.shape[B.ndim-batch_ndim-ndim:"
                "B.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B^T match strassen
static inline void strassen_check_AT_BT(const TensorTraits &A,
                                        const TensorTraits &B, Index ndim,
                                        Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[i] != B.shape[B.ndim - batch_ndim - ndim + i])
        {
            throw std::runtime_error(
                "A.shape[0:ndim] != "
                "B.shape[B.ndim-batch_ndim-ndim:B.ndim-batch_ndim]");
        }
        if(A.basetile_shape[i] !=
           B.basetile_shape[B.ndim - batch_ndim - ndim + i])
        {
            throw std::runtime_error("A.basetile_shape[0:ndim] != "
                                     "B.basetile_shape[B.ndim-batch_ndim-ndim:"
                                     "B.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and op(B) match strassen
static inline void strassen_check_opA_opB(const TransOp &transA,
                                          const TensorTraits &A,
                                          const TransOp &transB,
                                          const TensorTraits &B, Index ndim,
                                          Index batch_ndim)
{
    switch(transB.value)
    {
    case TransOp::NoTrans:
        switch(transA.value)
        {
        case TransOp::NoTrans:
            strassen_check_A_B(A, B, ndim, batch_ndim);
            break;
        case TransOp::Trans:
            strassen_check_AT_B(A, B, ndim);
            break;
        default:
            throw std::runtime_error("Wrong value of transA");
        }
        break;
    case TransOp::Trans:
        switch(transA.value)
        {
        case TransOp::NoTrans:
            strassen_check_A_BT(A, B, ndim, batch_ndim);
            break;
        case TransOp::Trans:
            strassen_check_AT_BT(A, B, ndim, batch_ndim);
            break;
        default:
            throw std::runtime_error("Wrong value of transA");
        }
        break;
    default:
        throw std::runtime_error("Wrong value of transB");
    }
}

//! Check if shapes of tensors A and C match strassen
static inline void strassen_check_A_C(const TensorTraits &A,
                                      const TensorTraits &C, Index ndim,
                                      Index batch_ndim)
{
    for(Index i = 0; i < A.ndim - batch_ndim - ndim; ++i)
    {
        if(A.shape[i] != C.shape[i])
        {
            throw std::runtime_error("A.shape[0:A.ndim-batch_ndim-ndim] != "
                                     "C.shape[0:A.ndim-batch_ndim-ndim]");
        }
        if(A.basetile_shape[i] != C.basetile_shape[i])
        {
            throw std::runtime_error(
                "A.basetile_shape[0:"
                "A.ndim-batch_ndim-ndim] != "
                "C.basetile_shape[0:A.ndim-batch_ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and C match strassen
static inline void strassen_check_AT_C(const TensorTraits &A,
                                       const TensorTraits &C, Index ndim,
                                       Index batch_ndim)
{
    for(Index i = ndim; i < A.ndim - batch_ndim; ++i)
    {
        if(A.shape[i] != C.shape[i - ndim])
        {
            throw std::runtime_error("A.shape[ndim:A.ndim-batch_ndim] != "
                                     "C.shape[0:A.ndim-batch_ndim-ndim]");
        }
        if(A.basetile_shape[i] != C.basetile_shape[i - ndim])
        {
            throw std::runtime_error(
                "A.basetile_shape[ndim:"
                "A.ndim-batch_ndim] != "
                "C.basetile_shape[0:A.ndim-batch_ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and C match strassen
static inline void strassen_check_opA_C(const TransOp &transA,
                                        const TensorTraits &A,
                                        const TensorTraits &C, Index ndim,
                                        Index batch_ndim)
{
    switch(transA.value)
    {
    case TransOp::NoTrans:
        strassen_check_A_C(A, C, ndim, batch_ndim);
        break;
    case TransOp::Trans:
        strassen_check_AT_C(A, C, ndim, batch_ndim);
        break;
        // This parameter was already checked in strassen_check_opA_opB
    }
}

//! Check if shapes of tensors B and C match strassen
static inline void strassen_check_B_C(const TensorTraits &B,
                                      const TensorTraits &C, Index ndim,
                                      Index batch_ndim)
{
    for(Index i = ndim; i < B.ndim - batch_ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim - B.ndim + i])
        {
            throw std::runtime_error(
                "B.shape[ndim:B.ndim-batch_ndim] != "
                "C.shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
        if(B.basetile_shape[i] != C.basetile_shape[C.ndim - B.ndim + i])
        {
            throw std::runtime_error(
                "B.basetile_shape[ndim:"
                "B.ndim-batch_ndim] != "
                "C.basetile_shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors B^T and C match strassen
static inline void strassen_check_BT_C(const TensorTraits &B,
                                       const TensorTraits &C, Index ndim,
                                       Index batch_ndim)
{
    for(Index i = 0; i < B.ndim - batch_ndim - ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim - B.ndim + ndim + i])
        {
            throw std::runtime_error(
                "B.shape[0:B.ndim-batch_ndim-ndim] != "
                "C.shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
        if(B.basetile_shape[i] != C.basetile_shape[C.ndim - B.ndim + ndim + i])
        {
            throw std::runtime_error(
                "B.basetile_shape[0:"
                "B.ndim-batch_ndim-ndim] != "
                "C.basetile_shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors op(B) and C match strassen
static inline void strassen_check_opB_C(const TransOp &transB,
                                        const TensorTraits &B,
                                        const TensorTraits &C, Index ndim,
                                        Index batch_ndim)
{
    switch(transB.value)
    {
    case TransOp::NoTrans:
        strassen_check_B_C(B, C, ndim, batch_ndim);
        break;
    case TransOp::Trans:
        strassen_check_BT_C(B, C, ndim, batch_ndim);
        break;
        // This parameter was already checked in strassen_check_opA_opB
    }
}

//! Check if tile layout allow the use of strassen
static inline void strassen_check_leftover_tiles(const TensorTraits &A,
                                                 const TensorTraits &B,
                                                 const TensorTraits &C)
{
    // Check if ndim is negative since it will be converted to Index
    for(Index i = 0; i < 2; i++)
    {
        if(A.leftover_shape[i] != A.basetile_shape[i])
        {
            throw std::runtime_error("A.leftover_shape != A.basetile_shape");
        }
        if(B.leftover_shape[i] != B.basetile_shape[i])
        {
            throw std::runtime_error("B.leftover_shape != B.basetile_shape");
        }
    }
}

//! Check if tensors match strassen
void strassen_check(const TransOp &transA, const TensorTraits &A,
                    const TransOp &transB, const TensorTraits &B,
                    const TensorTraits &C, Index ndim, Index batch_ndim)
{
    // Check if dimensionalities match
    strassen_check_ndim(A, B, C, ndim, batch_ndim);
    // Check if batch shapes match
    strassen_check_batch(A, B, C, batch_ndim);
    // Check if shapes of A and B match
    strassen_check_opA_opB(transA, A, transB, B, ndim, batch_ndim);
    // Check if shapes of A and C match
    strassen_check_opA_C(transA, A, C, ndim, batch_ndim);
    // Check if shapes of B and C match
    strassen_check_opB_C(transB, B, C, ndim, batch_ndim);
    // Check leftover tiles
    strassen_check_leftover_tiles(A, B, C);
}

//! Allocate 7x tiles for A, B and C as temporary memory
template <typename T>
std::vector<std::vector<std::vector<std::vector<starpu::VariableHandle>>>>
allocate_memory_for_quarters(size_t size, Index s1, Index s2, Index s3,
                             Index s4)
{
    std::vector<std::vector<std::vector<std::vector<starpu::VariableHandle>>>>
        ret;
    // First index - Strassen's, i.e. 1-7 (0 ignored)
    for(int i1 = 0; i1 <= s1; i1++)
    {
        ret.push_back(
            std::vector<std::vector<std::vector<starpu::VariableHandle>>>());
        if(i1 == 0)
            continue;
        // Second index - batch number
        for(int i2 = 0; i2 < s2; i2++)
        {
            ret[i1].push_back(
                std::vector<std::vector<starpu::VariableHandle>>());
            // Third index - one of the axes
            for(int i3 = 0; i3 < s3; i3++)
            {
                ret[i1][i2].push_back(std::vector<starpu::VariableHandle>());
                // Forth index - one of the axes
                for(int i4 = 0; i4 < s4; i4++)
                {
                    ret[i1][i2][i3].push_back(
                        starpu::VariableHandle(size, STARPU_RW));
                }
            }
        }
    }
    return ret;
}

//! Calculate weighted sum of chosen quaters for Strassen
template <typename T, int Ij1, int iJ1, int Ij2, int iJ2, int weight>
void _calculate_quater(
    Index batch, Index tile_batch, Index k, Index tile_k, Index m, Index tile_m,
    const TransOp &transA, const Tensor<T> &A,
    std::vector<std::vector<std::vector<starpu::VariableHandle>>> &quater)
{
    constexpr Scalar one = 1.0, zero = 0.0, w = weight;

    std::array<Index, 2> opA_stride;
    switch(transA.value)
    {
    case TransOp::NoTrans:
        opA_stride = {1, m};
        break;
    case TransOp::Trans:
        opA_stride = {k, 1};
        break;
    }

    // All per-tile starpu strassen calls shall appear here
    for(Index b = 0; b < batch; ++b)
    {
        for(Index j = 0; j < (k + 1) / 2; ++j)
        {
            for(Index i = 0; i < (m + 1) / 2; ++i)
            {
                bool outside_i = (i == m / 2) && (m % 2 == 1);
                bool outside_j = (j == k / 2) && (k % 2 == 1);
                Index base_A_tile_offset =
                    opA_stride[0] * i + opA_stride[1] * j + b * m * k;

                if(outside_i && Ij1 == 2 || outside_j && iJ1 == 2)
                {
                    auto tile_handle = A.get_tile_handle(0);
                    starpu::clear::submit(quater[b][i][j]);
                }
                else
                {
                    Index tile_offset =
                        base_A_tile_offset +
                        (Ij1 - 1) * opA_stride[0] * ((m + 1) / 2) +
                        (iJ1 - 1) * opA_stride[1] * ((k + 1) / 2);
                    auto tile_handle = A.get_tile_handle(tile_offset);

                    // Execute
                    starpu::add::submit<T>(tile_m * tile_k * tile_batch, one,
                                           tile_handle, zero, quater[b][i][j]);
                }

                if(weight == 0 || outside_i && Ij2 == 2 ||
                   outside_j && iJ2 == 2)
                    continue;

                Index tile_offset = base_A_tile_offset +
                                    (Ij2 - 1) * opA_stride[0] * ((m + 1) / 2) +
                                    (iJ2 - 1) * opA_stride[1] * ((k + 1) / 2);
                auto tile_handle = A.get_tile_handle(tile_offset);
                starpu::add::submit<T>(tile_m * tile_k * tile_batch, w,
                                       tile_handle, one, quater[b][i][j]);

                // Flush cache for the output tile on every node
                quater[b][m][k].mpi_flush();
            }
        }
    }
}

//! Calculate sum of chosen quaters for Strassen
template <typename T, int Ij1, int iJ1, int Ij2, int iJ2>
void calculate_quater_sum(
    Index batch, Index tile_batch, Index k, Index tile_k, Index m, Index tile_m,
    const TransOp &transA, const Tensor<T> &A,
    std::vector<std::vector<std::vector<starpu::VariableHandle>>> &quater)
{
    return _calculate_quater<T, Ij1, iJ1, Ij2, iJ2, 1>(
        batch, tile_batch, k, tile_k, m, tile_m, transA, A, quater);
}

//! Calculate difference of chosen quaters for Strassen
template <typename T, int Ij1, int iJ1, int Ij2, int iJ2>
void calculate_quater_sub(
    Index batch, Index tile_batch, Index k, Index tile_k, Index m, Index tile_m,
    const TransOp &transA, const Tensor<T> &A,
    std::vector<std::vector<std::vector<starpu::VariableHandle>>> &quater)
{
    return _calculate_quater<T, Ij1, iJ1, Ij2, iJ2, -1>(
        batch, tile_batch, k, tile_k, m, tile_m, transA, A, quater);
}

//! Copy chosen quater for Strassen
template <typename T, int Ij, int iJ>
void calculate_quater(
    Index batch, Index tile_batch, Index k, Index tile_k, Index m, Index tile_m,
    const TransOp &transA, const Tensor<T> &A,
    std::vector<std::vector<std::vector<starpu::VariableHandle>>> &quater)
{
    return _calculate_quater<T, Ij, iJ, Ij, iJ, 0>(
        batch, tile_batch, k, tile_k, m, tile_m, transA, A, quater);
}

//! Asynchronous version of tensor-wise strassen operation
/*! Matrix multiplication for tensors, which are virtually reshaped
 *
 * @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tensor A
 * @param[in] A: Input tensor A
 * @param[in] transB: Transposition flag for the tensor B
 * @param[in] B: Input tensor B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tensor C
 * @param[in] ndim: Number of dimensions used in strassen contraction
 * @param[in] batch_ndim: Number of last dimensions used for batching of
 * strassens
 * @param[in] redux: Whether or not to use STARPU_REDUX
 * */
template <typename T>
void strassen_async(Scalar alpha, const TransOp &transA, const Tensor<T> &A,
                    const TransOp &transB, const Tensor<T> &B, Scalar beta,
                    const Tensor<T> &C, Index ndim, Index batch_ndim, int redux)
{

    // Check inputs (throw exception in case of an error)
    strassen_check(transA, A, transB, B, C, ndim, batch_ndim);
    // Sizes of A, B and C as simple matrices (grids of tiles) for strassen
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    constexpr Scalar one = 1.0, zero = 0.0;
    Index m = C.grid.matrix_shape[A.ndim - batch_ndim - ndim][0];
    Index batch = C.grid.matrix_shape[C.ndim - batch_ndim][1];
    Index n = C.grid.matrix_shape[A.ndim - batch_ndim - ndim][1] / batch;
    Index k;
    std::array<Index, 2> opA_stride, opB_stride;
    switch(transA.value)
    {
    case TransOp::NoTrans:
        k = A.grid.matrix_shape[A.ndim - batch_ndim - ndim][1] / batch;
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

    auto C_tile_traits = C.get_tile_traits(0);
    // Getting sizes
    Index tile_m = C_tile_traits.matrix_shape[A.ndim - batch_ndim - ndim][0];
    Index tile_batch = C_tile_traits.matrix_shape[C.ndim - batch_ndim][1];
    Index tile_n =
        C_tile_traits.matrix_shape[A.ndim - batch_ndim - ndim][1] / tile_batch;
    Index tile_k;
    auto A_first_tile_traits = A.get_tile_traits(0);
    switch(transA.value)
    {
    case TransOp::NoTrans:
        tile_k =
            A_first_tile_traits.matrix_shape[A.ndim - batch_ndim - ndim][1] /
            tile_batch;
        break;
        // This parameter was already checked
        // case TransOp::Trans:
    default:
        tile_k = A_first_tile_traits.matrix_shape[ndim][0];
        break;
    }

    const size_t A_size = tile_m * tile_k * tile_batch * sizeof(T);
    auto A_tmp = allocate_memory_for_quarters<T>(A_size, 7, batch, (m + 1) / 2,
                                                 (k + 1) / 2);
    const size_t B_size = tile_n * tile_k * tile_batch * sizeof(T);
    auto B_tmp = allocate_memory_for_quarters<T>(B_size, 7, batch, (k + 1) / 2,
                                                 (n + 1) / 2);
    const size_t C_size = tile_n * tile_m * tile_batch * sizeof(T);
    auto C_tmp = allocate_memory_for_quarters<T>(C_size, 7, batch, (m + 1) / 2,
                                                 (n + 1) / 2);

    calculate_quater_sum<T, 1, 1, 2, 2>(batch, tile_batch, k, tile_k, m, tile_m,
                                        transA, A, A_tmp[1]);
    calculate_quater_sum<T, 2, 1, 2, 2>(batch, tile_batch, k, tile_k, m, tile_m,
                                        transA, A, A_tmp[2]);
    calculate_quater<T, 1, 1>(batch, tile_batch, k, tile_k, m, tile_m, transA,
                              A, A_tmp[3]);
    calculate_quater<T, 2, 2>(batch, tile_batch, k, tile_k, m, tile_m, transA,
                              A, A_tmp[4]);
    calculate_quater_sum<T, 1, 1, 1, 2>(batch, tile_batch, k, tile_k, m, tile_m,
                                        transA, A, A_tmp[5]);
    calculate_quater_sub<T, 2, 1, 1, 1>(batch, tile_batch, k, tile_k, m, tile_m,
                                        transA, A, A_tmp[6]);
    calculate_quater_sub<T, 1, 2, 2, 2>(batch, tile_batch, k, tile_k, m, tile_m,
                                        transA, A, A_tmp[7]);

    calculate_quater_sum<T, 1, 1, 2, 2>(batch, tile_batch, n, tile_n, k, tile_k,
                                        transB, B, B_tmp[1]);
    calculate_quater<T, 1, 1>(batch, tile_batch, n, tile_n, k, tile_k, transB,
                              B, B_tmp[2]);
    calculate_quater_sub<T, 1, 2, 2, 2>(batch, tile_batch, n, tile_n, k, tile_k,
                                        transB, B, B_tmp[3]);
    calculate_quater_sub<T, 2, 1, 1, 1>(batch, tile_batch, n, tile_n, k, tile_k,
                                        transB, B, B_tmp[4]);
    calculate_quater<T, 2, 2>(batch, tile_batch, n, tile_n, k, tile_k, transB,
                              B, B_tmp[5]);
    calculate_quater_sum<T, 1, 1, 1, 2>(batch, tile_batch, n, tile_n, k, tile_k,
                                        transB, B, B_tmp[6]);
    calculate_quater_sum<T, 2, 1, 2, 2>(batch, tile_batch, n, tile_n, k, tile_k,
                                        transB, B, B_tmp[7]);

    for(int s = 1; s <= 7; s++)
    {
        for(Index b = 0; b < batch; ++b)
        {
            for(Index j = 0; j < (n + 1) / 2; ++j)
            {
                for(Index i = 0; i < (m + 1) / 2; ++i)
                {
                    for(Index ij = 0; ij < (k + 1) / 2; ++ij)
                    {
                        starpu::gemm::submit<T>(
                            transA, transB, tile_m, tile_n, tile_k, tile_batch,
                            one, A_tmp[s][b][i][ij], B_tmp[s][b][ij][j],
                            ij == 0 ? zero : one, C_tmp[s][b][i][j], redux);
                    }
                }
            }
        }
    }

    for(Index b = 0; b < batch; ++b)
    {
        for(Index j = 0; j < (n + 1) / 2; ++j)
        {
            for(Index i = 0; i < (m + 1) / 2; ++i)
            {
                bool outside_i = (i == m / 2) && (m % 2 == 1);
                bool outside_j = (j == n / 2) && (n % 2 == 1);
                Index base_C_tile_offset = j * m + i + b * n * m;

                if(true)
                {
                    Index C_tile_offset = base_C_tile_offset;
                    auto C_tile_handle = C.get_tile_handle(C_tile_offset);
                    starpu::add::submit<T>(tile_n * tile_m * tile_batch, alpha,
                                           C_tmp[1][b][i][j], beta,
                                           C_tile_handle);
                    starpu::add::submit<T>(tile_n * tile_m * tile_batch, alpha,
                                           C_tmp[4][b][i][j], one,
                                           C_tile_handle);
                    starpu::add::submit<T>(tile_n * tile_m * tile_batch, -alpha,
                                           C_tmp[5][b][i][j], one,
                                           C_tile_handle);
                    starpu::add::submit<T>(tile_n * tile_m * tile_batch, alpha,
                                           C_tmp[7][b][i][j], one,
                                           C_tile_handle);
                }

                if(!outside_j)
                {
                    Index C_tile_offset =
                        base_C_tile_offset + ((n + 1) / 2) * m;
                    auto C_tile_handle = C.get_tile_handle(C_tile_offset);
                    starpu::add::submit<T>(tile_n * tile_m * tile_batch, alpha,
                                           C_tmp[3][b][i][j], beta,
                                           C_tile_handle);
                    starpu::add::submit<T>(tile_n * tile_m * tile_batch, alpha,
                                           C_tmp[5][b][i][j], one,
                                           C_tile_handle);
                }

                if(!outside_i)
                {
                    Index C_tile_offset = base_C_tile_offset + ((m + 1) / 2);
                    auto C_tile_handle = C.get_tile_handle(C_tile_offset);
                    starpu::add::submit<T>(tile_n * tile_m * tile_batch, alpha,
                                           C_tmp[2][b][i][j], beta,
                                           C_tile_handle);
                    starpu::add::submit<T>(tile_n * tile_m * tile_batch, alpha,
                                           C_tmp[4][b][i][j], one,
                                           C_tile_handle);
                }

                if(!outside_i && !outside_j)
                {
                    Index C_tile_offset =
                        base_C_tile_offset + ((n + 1) / 2) * m + ((m + 1) / 2);
                    auto C_tile_handle = C.get_tile_handle(C_tile_offset);
                    starpu::add::submit<T>(tile_n * tile_m * tile_batch, alpha,
                                           C_tmp[1][b][i][j], beta,
                                           C_tile_handle);
                    starpu::add::submit<T>(tile_n * tile_m * tile_batch, -alpha,
                                           C_tmp[2][b][i][j], one,
                                           C_tile_handle);
                    starpu::add::submit<T>(tile_n * tile_m * tile_batch, alpha,
                                           C_tmp[3][b][i][j], one,
                                           C_tile_handle);
                    starpu::add::submit<T>(tile_n * tile_m * tile_batch, alpha,
                                           C_tmp[6][b][i][j], one,
                                           C_tile_handle);
                }
            }
        }
    }
}

//! Blocking version of tensor-wise strassen operation
/*! Matrix multiplication for tensors, which are virtually reshaped
 *
 * @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tensor A
 * @param[in] A: Input tensor A
 * @param[in] transB: Transposition flag for the tensor B
 * @param[in] B: Input tensor B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tensor C
 * @param[in] ndim: Number of dimensions used in strassen contraction
 * @param[in] batch_ndim: Number of last dimensions used for batching of
 * strassens
 * */
template <typename T>
void strassen(Scalar alpha, const TransOp &transA, const Tensor<T> &A,
              const TransOp &transB, const Tensor<T> &B, Scalar beta,
              const Tensor<T> &C, Index ndim, Index batch_ndim, int redux)
{
    strassen_async<T>(alpha, transA, A, transB, B, beta, C, ndim, batch_ndim,
                      redux);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template void strassen_async<fp32_t>(Scalar alpha, const TransOp &transA,
                                     const Tensor<fp32_t> &A,
                                     const TransOp &transB,
                                     const Tensor<fp32_t> &B, Scalar beta,
                                     const Tensor<fp32_t> &C, Index ndim,
                                     Index batch_ndim, int redux);

template void strassen_async<fp64_t>(Scalar alpha, const TransOp &transA,
                                     const Tensor<fp64_t> &A,
                                     const TransOp &transB,
                                     const Tensor<fp64_t> &B, Scalar beta,
                                     const Tensor<fp64_t> &C, Index ndim,
                                     Index batch_ndim, int redux);

// template
// void strassen_async<fp16_t, fp32_t>(fp32_t alpha, const TransOp &transA,
//         const Tensor<fp16_t> &A,
//         const TransOp &transB, const Tensor<fp16_t> &B, fp32_t beta,
//         const Tensor<fp16_t> &C, Index ndim, Index batch_ndim, int redux);

// Explicit instantiation
template void strassen<fp32_t>(Scalar alpha, const TransOp &transA,
                               const Tensor<fp32_t> &A, const TransOp &transB,
                               const Tensor<fp32_t> &B, Scalar beta,
                               const Tensor<fp32_t> &C, Index ndim,
                               Index batch_ndim, int redux);

template void strassen<fp64_t>(Scalar alpha, const TransOp &transA,
                               const Tensor<fp64_t> &A, const TransOp &transB,
                               const Tensor<fp64_t> &B, Scalar beta,
                               const Tensor<fp64_t> &C, Index ndim,
                               Index batch_ndim, int redux);

// template
// void strassen<fp16_t, fp32_t>(fp32_t alpha, const TransOp &transA,
//         const Tensor<fp16_t> &A,
//         const TransOp &transB, const Tensor<fp16_t> &B, fp32_t beta,
//         const Tensor<fp16_t> &C, Index ndim, Index batch_ndim, int redux);

} // namespace tensor
} // namespace nntile
