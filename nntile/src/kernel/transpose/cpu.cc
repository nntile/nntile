/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/transpose/cpu.cc
 * Transpose operation on buffers on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/transpose/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::transpose
{

template<typename T>
void cpu(Index m, Index n, Scalar alpha_, const T* src, T* dst)
    noexcept
//! Transpose buffers on CPU
/*! dst[i,j] = alpha * src[j,i]
 *
 * @param[in] m: Number of rows of src and columns of dst
 * @param[in] n: Number of columns of src and rows of dst
 * @param[in] alpha_: Scalar multiplier
 * @param[in] src: Source tensor
 * @param[out] dst: Destination of the add operation
 * */
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    for(Index i = 0; i < m; ++i)
    {
        for(Index j = 0; j < n; ++j)
        {
            dst[i*n+j] = static_cast<T>(alpha * Y{src[i+j*m]});
        }
    }
}

//! Transpose an nr-by-nc sub-block of an m-by-n buffer on CPU
/*! Performs the following operation for di in [0, nr) and dj in [0, nc):
 *      dst[di*n+dj] = alpha * src[di+dj*m]
 *
 * src and dst are the sub-block's base pointers (its (0, 0) corner in
 * the whole m-by-n matrix); the sub-block KEEPS the whole matrix's
 * strides - the src columns stay m elements apart, the dst rows n
 * elements apart - so a sub-block is addressed exactly like a
 * standalone matrix whose leading dimensions are the whole buffer's.
 * The chunked NNHaul wrapper drives this entry with sub-blocks whose
 * covered src column segments and whose covered dst flat range each
 * lie inside single device chunks; the whole-buffer cpu() above is the
 * (m, n) case. Elementwise and deterministic: any piece factorization
 * is bitwise the whole-buffer call.
 *
 * @param[in] m: Size of the first mode of the whole src buffer.
 * @param[in] n: Size of the last mode of the whole src buffer.
 * @param[in] nr: Number of rows of the sub-block.
 * @param[in] nc: Number of columns of the sub-block.
 * @param[in] alpha: Scalar factor for src.
 * @param[in] src: Base pointer of the sub-block (m column stride).
 * @param[inout] dst: Base pointer of the sub-block (n row stride).
 * */
template<typename T>
void panel(Index m, Index n, Index nr, Index nc, Scalar alpha_,
    const T* src, T* dst)
    noexcept
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    for(Index di = 0; di < nr; ++di)
    {
        const T* src_ = src + di;
        T* dst_ = dst + di * n;
        for(Index dj = 0; dj < nc; ++dj)
        {
            dst_[dj] = static_cast<T>(alpha * Y{src_[dj * m]});
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Scalar alpha, const fp32_t* src,
        fp32_t* dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Scalar alpha, const fp64_t* src,
        fp64_t* dst)
    noexcept;

template
void panel<fp32_t>(Index m, Index n, Index nr, Index nc, Scalar alpha,
    const fp32_t* src, fp32_t* dst)
    noexcept;

template
void panel<fp64_t>(Index m, Index n, Index nr, Index nc, Scalar alpha,
    const fp64_t* src, fp64_t* dst)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Scalar alpha, const bf16_t* src,
        bf16_t* dst)
    noexcept;

template
void panel<bf16_t>(Index m, Index n, Index nr, Index nc, Scalar alpha,
    const bf16_t* src, bf16_t* dst)
    noexcept;

template
void cpu<fp16_t>(Index m, Index n, Scalar alpha, const fp16_t* src,
        fp16_t* dst)
    noexcept;

template
void panel<fp16_t>(Index m, Index n, Index nr, Index nc, Scalar alpha,
    const fp16_t* src, fp16_t* dst)
    noexcept;

} // namespace nntile::kernel::tranpose
