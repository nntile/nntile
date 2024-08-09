/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/addcdiv.cc
 * Addcdiv operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/addcdiv.hh"
#include "nntile/starpu/addcdiv.hh"

namespace nntile::tile
{

//! Asynchronous version of tile-wise addcdiv operation
/*! @param[in] val: Input scalar for sclaing of tile ratio
 * @param[in] eps: Input scalar for add to denominator
 * @param[in] nom: Input tile to store nominator values
 * @param[in] denom: Input scalar to store denominator values
 * @param[inout] src: Input and output tile for the addcdiv operation
 * */
template<typename T>
void addcdiv_async(Scalar val, Scalar eps, const Tile<T> &nom, const Tile<T> &denom, const Tile<T> &src)
{
    // Check shapes
    if(nom.shape != denom.shape)
    {
        throw std::runtime_error("Shapes of nominator and denominator are not equal");
    }
    if(nom.shape != src.shape)
    {
        throw std::runtime_error("Nominator shape is not equal to src shape");
    }
    // Submit task
    starpu::addcdiv::submit<T>(val, eps, src.nelems, nom, denom, src);
}

//! Blocking version of tile-wise addcdiv operation
/*! @param[in] val: Input scalar for sclaing of tile ratio
 * @param[in] eps: Input scalar for add to denominator
 * @param[in] nom: Input tile to store nominator values
 * @param[in] denom: Input scalar to store denominator values
 * @param[inout] src: Input and output tile for the addcdiv operation
 * */
template<typename T>
void addcdiv(Scalar val, Scalar eps, const Tile<T> &nom, const Tile<T> &denom, const Tile<T> &src)
{
    addcdiv_async<T>(val, eps, nom, denom, src);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void addcdiv_async<fp32_t>(Scalar val, Scalar eps, const Tile<fp32_t> &nom,
                        const Tile<fp32_t> &denom, const Tile<fp32_t> &src);

template
void addcdiv_async<fp32_fast_tf32_t>(Scalar val, Scalar eps, const Tile<fp32_fast_tf32_t> &nom,
                        const Tile<fp32_fast_tf32_t> &denom, const Tile<fp32_fast_tf32_t> &src);

template
void addcdiv_async<fp64_t>(Scalar val, Scalar eps, const Tile<fp64_t> &nom,
                        const Tile<fp64_t> &denom, const Tile<fp64_t> &src);

template
void addcdiv_async<bf16_t>(Scalar val, Scalar eps, const Tile<bf16_t> &nom,
                     const Tile<bf16_t> &denom, const Tile<bf16_t> &src);

// Explicit instantiation
template
void addcdiv<fp32_t>(Scalar val, Scalar eps, const Tile<fp32_t> &nom,
                     const Tile<fp32_t> &denom, const Tile<fp32_t> &src);

template
void addcdiv<fp32_fast_tf32_t>(Scalar val, Scalar eps, const Tile<fp32_fast_tf32_t> &nom,
                     const Tile<fp32_fast_tf32_t> &denom, const Tile<fp32_fast_tf32_t> &src);

template
void addcdiv<fp64_t>(Scalar val, Scalar eps, const Tile<fp64_t> &nom,
                     const Tile<fp64_t> &denom, const Tile<fp64_t> &src);

template
void addcdiv<bf16_t>(Scalar val, Scalar eps, const Tile<bf16_t> &nom,
                     const Tile<bf16_t> &denom, const Tile<bf16_t> &src);

} // namespace nntile::tile
