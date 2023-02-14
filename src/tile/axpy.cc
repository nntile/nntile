/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/axpy.cc
 * AXPY for two Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-14
 * */

#include "nntile/tile/axpy.hh"
#include "nntile/starpu/axpy.hh"

namespace nntile
{
namespace tile
{

//! Asynchronous version of tile-wise axpy operation
/*! @param[in] src: Input tile for element-wise axpy operation
 * @param[inout] dst: Input and output tile for the axpy operation
 * */
template<typename T>
void axpy_async(const Tile<T> &alpha, const Tile<T> &src, const Tile<T> &dst)
{
    // Check shapes
    if(alpha.shape.size() != 0)
    {
        throw std::runtime_error("alpha.shape.size() != 0");
    }
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Submit task
    starpu::axpy::submit<T>(alpha, src.nelems, src, dst);
}

//! Blocking version of tile-wise axpy operation
/*! @param[in] src: Input tile for element-wise axpy operation
 * @param[inout] dst: Input and output tile for the axpy operation
 * */
template<typename T>
void axpy(const Tile<T> &alpha, const Tile<T> &src, const Tile<T> &dst)
{
    axpy_async<T>(alpha, src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void axpy_async<fp32_t>(const Tile<fp32_t> &alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void axpy_async<fp64_t>(const Tile<fp64_t> &alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

// Explicit instantiation
template
void axpy<fp32_t>(const Tile<fp32_t> &alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void axpy<fp64_t>(const Tile<fp64_t> &alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

//! Asynchronous version of tile-wise axpy operation
/*! @param[in] src: Input tile for element-wise axpy operation
 * @param[inout] dst: Input and output tile for the axpy operation
 * */
template<typename T>
void axpy_async(T alpha, const Tile<T> &src, const Tile<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Submit task
    starpu::axpy::submit<T>(alpha, src.nelems, src, dst);
}

//! Blocking version of tile-wise axpy operation
/*! @param[in] src: Input tile for element-wise axpy operation
 * @param[inout] dst: Input and output tile for the axpy operation
 * */
template<typename T>
void axpy(T alpha, const Tile<T> &src, const Tile<T> &dst)
{
    axpy_async<T>(alpha, src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void axpy_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void axpy_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

// Explicit instantiation
template
void axpy<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void axpy<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

} // namespace tile
} // namespace nntile

