/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/sumnorm.hh
 * Sum and Euclidian norm of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include <nntile/tile/tile.hh>

namespace nntile
{

extern starpu_perfmodel sumnorm_perfmodel_fp32, sumnorm_perfmodel_fp64;

extern StarpuCodelet sumnorm_codelet_fp32, sumnorm_codelet_fp64;

template<typename T>
constexpr StarpuCodelet *sumnorm_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *sumnorm_codelet<fp32_t>()
{
    return &sumnorm_codelet_fp32;
}

template<>
constexpr StarpuCodelet *sumnorm_codelet<fp64_t>()
{
    return &sumnorm_codelet_fp64;
}

//! Tile-wise sum and scaled sum of squares along single given axis
//
// Main computational routine that does NO argument checking.
// @param[in] src: Source tile to get mean and variance
// @param[out] sum_ssq: Sum and scaled sum of squares along given axis
// @param[in] axis: Axis to be used
template<typename T>
void sumnorm_work(const Tile<T> &src, const Tile<T> &sumnorm, Index axis);

//! Tile-wise sum and scaled sum of squares along single given axis
//
// Checks input arguments
template<typename T>
void sumnorm_async(const Tile<T> &src, const Tile<T> &sumnorm, Index axis)
{
    // Check dimensions
    if(src.ndim != sumnorm.ndim)
    {
        throw std::runtime_error("src.ndim != sumnorm.ndim");
    }
    // Treat special case of src.ndim=0
    if(src.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= src.ndim)
    {
        throw std::runtime_error("axis >= src.ndim");
    }
    // Check shapes of src and sumnorm
    if(sumnorm.shape[0] != 2)
    {
        throw std::runtime_error("sumnorm.shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(src.shape[i] != sumnorm.shape[i+1])
        {
            throw std::runtime_error("src.shape[i] != sumnorm.shape[i+1]");
        }
    }
    for(Index i = axis+1; i < src.ndim; ++i)
    {
        if(src.shape[i] != sumnorm.shape[i])
        {
            throw std::runtime_error("src.shape[i] != sumnorm.shape[i]");
        }
    }
    // Launch codelet
    sumnorm_work<T>(src, sumnorm, axis);
}

//! Tile-wise sum and scaled sum of squares along single given axis
//
// Checks input arguments and blocks until finished
template<typename T>
void sumnorm(const Tile<T> &src, const Tile<T> &sum_ssq, Index axis)
{
    sumnorm_async<T>(src, sum_ssq, axis);
    starpu_task_wait_for_all();
}

} // namespace nntile

