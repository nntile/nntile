/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/copy.cc
 * Copy StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/copy.hh"

// Standard libraries
#include <cstring>

namespace nntile::haul
{

//! Constructor
Copy::Copy():
    codelet("nntile_copy", &Copy::cpu, nullptr, nullptr)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Copy StarPU buffers on CPU
void Copy::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const void *src = ::nntile::haul::buf_as<void>(buffers, 0);
    void *dst = ::nntile::haul::buf_as<void>(buffers, 1);
    // Launch kernel
    std::memcpy(dst, src, args->nbytes);
}


void Copy::submit(int starpu_worker_hint, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
//! Insert copy task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Get arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nbytes = src.nbytes();
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

//! Copy operation object
Copy copy;

} // namespace nntile::haul
