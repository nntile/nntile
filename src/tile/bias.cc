/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/bias.cc
 * Bias operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-05
 * */

#include "nntile/tile/bias.hh"
#include "nntile/starpu/bias.hh"

namespace nntile
{

// Bias operation over single axis
template<typename T>
void bias_work(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    // Reshape inputs for simplicity: src -> (m,n), dst -> (m,k,n)
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = src.nelems;
        k = dst.shape[0];
    }
    else if(axis == dst.ndim-1)
    {
        m = src.nelems;
        n = 1;
        k = dst.shape[axis];
    }
    else
    {
        m = dst.stride[axis];
        n = dst.matrix_shape[axis+1][1];
        k = dst.shape[axis];
    }
    // Insert corresponding task
    nntile::starpu::bias<T>(m, n, k, src, dst);
}

// Explicit instantiation of template
template
void bias_work(const Tile<fp32_t> &src, const Tile<fp32_t> &dst, Index axis);

// Explicit instantiation of template
template
void bias_work(const Tile<fp64_t> &src, const Tile<fp64_t> &dst, Index axis);

} // namespace nntile

