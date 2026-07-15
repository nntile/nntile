/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core/maxsumexp.hh
 * Sum and Euclidean norm of Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

//! beta=0 overwrite dst; beta=1 accumulate into dst
template<typename T>
void maxsumexp_async(int starpu_worker_hint, const Tile<T> &src,
        const Tile<T> &dst, Index axis, Scalar beta, int redux=0);

template<typename T>
void maxsumexp(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst,
        Index axis, Scalar beta, int redux=0);

} // namespace nntile::core
