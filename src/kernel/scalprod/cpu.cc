/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/scalprod/cpu.cc
 * Scalar product of buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-03-26
 * */

#include "nntile/kernel/scalprod/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace scalprod
{

template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src1, const T *src2,
        T beta, T *dst)
    noexcept
//! Scalar product along middle axis
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute scalar
 * products of slices along second axis with k elements, resulting in m-by-n
 * output array dst. If beta is non-zero, then values of array dst are updated
 * by this routine in read-write mode, therefore dst must be initialized before
 * use. If beta is zero, all values of dst are overwritten and it shall be used
 * in write mode.
 *
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = beta*dst[i,j] + alpha*src1[i,:,j]@src2[i,:,j]
 *      
 * @param[in] m: Size of the first mode of src1, src2 and dst
 * @param[in] n: Size of the last mode of src1, src2 and dst
 * @param[in] k: Size of the middle mode of src1 and src2 arrays
 * @param[in] alpha: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-n array, that accumulates
 *      scalar products of src1 and src2 along middle axis.
 * */
{
    const Index mk = m * k;
    Index dst_offset = 0;
    constexpr T zero = 0;
    // Cycle over row of output buffer
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over column of output buffer
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Get scalar product of corresponding slices
            const T *src1_slice = src1 + i2*mk + i1;
            const T *src2_slice = src2 + i2*mk + i1;
            // Init scalar product
            T sum = 0.0;
            // Cycle over slices of inputs
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from sources
                T val1 = src1_slice[i0*m];
                T val2 = src2_slice[i0*m];
                // Accumulate scalar product
                sum += val1 * val2;
            }
            // Save result 
            if(beta == 0.0)
            {
                dst[dst_offset] = alpha * sum;
            }
            else
            {
                dst[dst_offset] = beta*dst[dst_offset] + alpha*sum;
            }
            // Cycle to next output element
            ++dst_offset;
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, fp32_t alpha, const fp32_t *src1,
        const fp32_t *src2, fp32_t beta, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, fp64_t alpha, const fp64_t *src1,
        const fp64_t *src2, fp64_t beta, fp64_t *dst)
    noexcept;

} // namespace scalprod
} // namespace kernel
} // namespace nntile

