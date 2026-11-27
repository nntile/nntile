/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/adamw_step.cc
 * AdamW operation of NNHaul buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/adamw_step.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/adamw_step.hh"
#include "nntile/nnhaul/chunk_units.hh"

//! NNHaul wrappers for one step of AdamW optimizer
namespace nntile::nnhaul
{

//! Constructor
template<typename T>
AdamWStep<std::tuple<T>>::AdamWStep():
    codelet(
        "nntile_adamw_step",
        &AdamWStep<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &AdamWStep<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &AdamWStep<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply Adam step on NNHaul buffers on CPU
template<typename T>
void AdamWStep<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto grad = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    auto first_moments = ::nntile::nnhaul::buf_chunks<T>(buffers, 1);
    auto second_moments = ::nntile::nnhaul::buf_chunks<T>(buffers, 2);
    auto p = ::nntile::nnhaul::buf_chunks<T>(buffers, 3);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
        args->num_elems,
        [&](std::size_t n, const T *g, T *f, T *s, T *pp) {
            kernel::adamw_step::cpu<T>(
                args->num_iter,
                n,
                args->beta_1,
                args->beta_2,
                args->eps,
                args->lr,
                args->weight_decay,
                g,
                f,
                s,
                pp
            );
        },
        grad,
        first_moments,
        second_moments,
        p);
}

// Specializations of CPU wrapper for accelerated types
template<>
void AdamWStep<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamWStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AdamWStep<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamWStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AdamWStep<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamWStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply Adam step operation on NNHaul buffer on CUDA
template<typename T>
void AdamWStep<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    chunkunits::require_kernel_index_envelope(
        static_cast<std::size_t>(args->num_elems), "nntile_adamw_step");
    // Get interfaces
    auto *grad_buf = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto *first_buf = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto *second_buf = static_cast<::nnhaul::Buffer *>(buffers[2]);
    auto *p_buf = static_cast<::nnhaul::Buffer *>(buffers[3]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if(grad_buf->chunk_count() == 1 && first_buf->chunk_count() == 1
            && second_buf->chunk_count() == 1 && p_buf->chunk_count() == 1)
    {
        if(args->num_elems > 0)
        {
            kernel::adamw_step::cuda<T>(stream, args->num_iter,
                args->num_elems, args->beta_1, args->beta_2, args->eps,
                args->lr, args->weight_decay, grad_buf->get_ptr<T>(),
                first_buf->get_ptr<T>(), second_buf->get_ptr<T>(),
                p_buf->get_ptr<T>());
        }
        return;
    }
    if(args->num_elems > 0)
    {
        chunkunits::ChunkView<T> const cvg(grad_buf);
        chunkunits::ChunkView<T> const cvf(first_buf);
        chunkunits::ChunkView<T> const cvs(second_buf);
        chunkunits::ChunkView<T> const cvp(p_buf);
        chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
        tabs.clear();
        std::size_t const o_pg = tabs.add(cvg);
        std::size_t const o_pf = tabs.add(cvf);
        std::size_t const o_ps = tabs.add(cvs);
        std::size_t const o_pp = tabs.add(cvp);
        chunkunits::ChunkPtrTable &arena = chunkunits::chunk_ptr_table();
        if(arena.begin(stream))
        {
            void const *table_base = arena.upload_tables(tabs);
            if(table_base != nullptr)
            {
                kernel::adamw_step::chunked<T>(stream, args->num_iter,
                    args->num_elems, args->beta_1, args->beta_2, args->eps,
                    args->lr, args->weight_decay,
                    arena.table_view<const T *>(table_base, o_pg),
                    Index(cvg.nchunks),
                    arena.table_view<T *>(table_base, o_pf),
                    Index(cvf.nchunks),
                    arena.table_view<T *>(table_base, o_ps),
                    Index(cvs.nchunks),
                    arena.table_view<T *>(table_base, o_pp),
                    Index(cvp.nchunks),
                    Index(cvg.chunk_elems), Index(cvf.chunk_elems),
                    Index(cvs.chunk_elems), Index(cvp.chunk_elems));
                arena.end(stream);
                return;
            }
            std::fprintf(stderr,
                "nntile_adamw_step: chunk-ptr table cannot serve this "
                "launch\n");
            std::abort();
        }
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void AdamWStep<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamWStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AdamWStep<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamWStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AdamWStep<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamWStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for adam_step tasks that depends only on cl_arg
template<typename T>
std::uint64_t AdamWStep<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::nnhaul::fnv1a(&args->num_elems, sizeof(args->num_elems), hash);
    return hash;
}

//! Submit AdamW step task
template<typename T>
void AdamWStep<std::tuple<T>>::submit(int worker_hint,
    Index num_iter,
    Index num_elems,
    Scalar beta_1,
    Scalar beta_2,
    Scalar eps,
    Scalar lr,
    Scalar weight_decay,
    ::nnhaul::Handle & grad,
    ::nnhaul::Handle & first_moment,
    ::nnhaul::Handle & second_moment,
    ::nnhaul::Handle & param
)
{
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->num_iter = num_iter;
    args->num_elems = num_elems;
    args->beta_1 = beta_1;
    args->beta_2 = beta_2;
    args->eps = eps;
    args->lr = lr;
    args->weight_decay = weight_decay;
    //double nflops = 5 * nelems;
    // Submit task
    enum data_access_mode moments_mode;
    if (num_iter == 1)
    {
        moments_mode = ACCESS_W;
    }
    else
    {
        moments_mode = ACCESS_RW;
    }
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &grad }, { moments_mode, &first_moment }, { moments_mode, &second_moment }, { ACCESS_RW, &param } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AdamWStep<std::tuple<nntile::fp64_t>>;
template class AdamWStep<std::tuple<nntile::fp32_t>>;
template class AdamWStep<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AdamWStep<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AdamWStep<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AdamWStep<std::tuple<nntile::bf16_t>>;
template class AdamWStep<std::tuple<nntile::fp16_t>>;

//! Pack of adamw_step operations for different types
adamw_step_pack_t adamw_step;

} // namespace nntile::nnhaul
