/*! @copyright (c) 2022-present Skolkovo Institute 
 *                              of Science and Technology (Skoltech), Russia.
 *                 2023-present Artificial Intelligence Research 
 *                              Institute (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/lion_step.cc
 * Per-element Lion optimizer step on StarPU buffers
 *
 * @version 1.1.0
 */

#ifndef STARPU_SIMGRID
#include "nntile/kernel/lion_step.hh"  
#endif // STARPU_SIMGRID

#include "nntile/starpu/lion_step.hh"  
#include <cstdlib> 

//! StarPU wrappers for one step of Lion optimizer
namespace nntile::starpu::lion_step
{


//-------------------------------------------
// 2) CPU Implementation
//-------------------------------------------
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run only if not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);

    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *grad          = interfaces[0]->get_ptr<T>();
    T *first_moment  = interfaces[1]->get_ptr<T>();
    T *p             = interfaces[2]->get_ptr<T>();

    // Launch Lion kernel on CPU
    kernel::lion_step::cpu<T>(
        args->num_iter, 
        args->num_elems,
        args->beta_1_,
        args->beta_2_,
        args->lambda_,
        args->lr_,
        args->weight_decay_,
        grad,
        first_moment,
        p
    );
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//-------------------------------------------
// 3) CUDA Implementation
//-------------------------------------------
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);

    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *grad          = interfaces[0]->get_ptr<T>();
    T *first_moment  = interfaces[1]->get_ptr<T>();
    T *p             = interfaces[2]->get_ptr<T>();

    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();

    // Launch Lion kernel on GPU
    kernel::lion_step::cuda<T>(
        stream,
        args->num_iter,
        args->num_elems,
        args->beta_1,
        args->beta_2,
        args->lambda_,
        args->lr,
        args->weight_decay,
        grad,
        first_moment,
        p
    );
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//-------------------------------------------
// 4) Codelets for different data types
//-------------------------------------------
Codelet codelet_fp32, codelet_fp64, codelet_fp32_fast_tf32, codelet_bf16;

//-------------------------------------------
// 5) Init function to set up codelets
//-------------------------------------------
void init()
{
    codelet_fp32.init("nntile_lion_step_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else
            {}
#endif
    );

    codelet_bf16.init("nntile_lion_step_bf16",
            nullptr,
            {cpu<bf16_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<bf16_t>}
#else
            {}
#endif
    );

    codelet_fp32_fast_tf32.init("nntile_lion_step_fp32_fast_tf32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else
            {}
#endif
    );

    codelet_fp64.init("nntile_lion_step_fp64",
            nullptr,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp64_t>}
#else
            {}
#endif
    );
}

//-------------------------------------------
// 6) Restrict / Restore Codelet Where
//-------------------------------------------
void restrict_where(uint32_t where)
{
    codelet_fp32.restrict_where(where);
    codelet_bf16.restrict_where(where);
    codelet_fp32_fast_tf32.restrict_where(where);
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_bf16.restore_where();
    codelet_fp32_fast_tf32.restore_where();
    codelet_fp64.restore_where();
}

//-------------------------------------------
// 7) Submit function to enqueue a Lion step
//-------------------------------------------
template<typename T>
void submit(Index num_iter, Index num_elems,
            Scalar beta_1, Scalar beta_2,
            Scalar lambda_, Scalar lr, Scalar weight_decay,
            Handle grad, Handle first_moment, Handle p)
{
    // Allocate codelet arguments
    args_t* args = reinterpret_cast<args_t*>(std::malloc(sizeof(*args)));
    args->num_iter      = num_iter;
    args->num_elems     = num_elems;
    args->beta_1_        = beta_1;
    args->beta_2_        = beta_2;
    args->lambda_       = lambda_;
    args->lr_            = lr;
    args->weight_decay_  = weight_decay;

    // If first iteration, we can treat momentum buffer as W(rite), otherwise RW
    enum starpu_data_access_mode moments_mode;
    if (num_iter == 1)
        moments_mode = STARPU_W;
    else
        moments_mode = STARPU_RW;

    // Insert StarPU task
    int ret = starpu_task_insert(codelet<T>(),
        STARPU_R,  static_cast<starpu_data_handle_t>(grad),
        moments_mode, static_cast<starpu_data_handle_t>(first_moment),
        STARPU_RW, static_cast<starpu_data_handle_t>(p),
        STARPU_CL_ARGS, args, sizeof(*args),
        0
    );

    if(ret != 0)
    {
        // std::free(args);
        throw std::runtime_error("Error in lion_step task submission");
    }
}

//-------------------------------------------
// 8) Explicit template instantiations
//-------------------------------------------
template
void submit<fp32_t>(Index num_iter, Index num_elems,
    Scalar beta_1, Scalar beta_2, Scalar lambda_, Scalar lr,
    Scalar weight_decay, Handle grad, Handle first_moment, Handle p);

template
void submit<fp32_fast_tf32_t>(Index num_iter, Index num_elems,
    Scalar beta_1, Scalar beta_2, Scalar lambda_, Scalar lr,
    Scalar weight_decay, Handle grad, Handle first_moment, Handle p);

template
void submit<fp64_t>(Index num_iter, Index num_elems,
    Scalar beta_1, Scalar beta_2, Scalar lambda_, Scalar lr,
    Scalar weight_decay, Handle grad, Handle first_moment, Handle p);

template
void submit<bf16_t>(Index num_iter, Index num_elems,
    Scalar beta_1, Scalar beta_2, Scalar lambda_, Scalar lr,
    Scalar weight_decay, Handle grad, Handle first_moment, Handle p);

} // namespace nntile::starpu::lion_step
