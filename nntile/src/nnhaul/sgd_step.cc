/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                 (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sgd_step.cc
 * Fused SGD with momentum step operation of StarPU buffers
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/sgd_step.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/sgd_step.hh"
#include "nntile/nnhaul/chunk_units.hh"

//! StarPU wrappers for one step of SGD with momentum optimizer
namespace nntile::haul
{

//! Constructor
template<typename T>
SGDStep<std::tuple<T>>::SGDStep():
    codelet(
        "nntile_sgd_step",
        &SGDStep<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &SGDStep<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &SGDStep<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply SGD step on StarPU buffers on CPU
template<typename T>
void SGDStep<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto grad = ::nntile::haul::buf_chunks<T>(buffers, 0);
    auto velocity = ::nntile::haul::buf_chunks<T>(buffers, 1);
    auto p = ::nntile::haul::buf_chunks<T>(buffers, 2);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::haul::for_each_flat_piece(
        args->num_elems,
        [&](std::size_t n, const T *g, T *v, T *pp) {
            kernel::sgd_step::cpu<T>(
                args->num_iter,
                n,
                args->momentum,
                args->lr,
                args->weight_decay,
                args->dampening,
                args->nesterov,
                g,
                v,
                pp
            );
        },
        grad,
        velocity,
        p);
}

// Specializations of CPU wrapper for accelerated types
template<>
void SGDStep<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SGDStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SGDStep<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SGDStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SGDStep<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SGDStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply SGD step operation on StarPU buffer on CUDA
template<typename T>
void SGDStep<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_grad = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_vel = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_p = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_grad->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_vel->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
        && (buf_p->chunk_count() == 1 && !chunkunits::force_chunked_kernels()))
    {
        // Classic contiguous dispatch: illegal under the force flag (the
        // tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_sgd_step", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::sgd_step::cuda<T>(
            stream,
            args->num_iter,
            args->num_elems,
            args->momentum,
            args->lr,
            args->weight_decay,
            args->dampening,
            args->nesterov,
            buf_grad->get_ptr<T>(),
            buf_vel->get_ptr<T>(),
            buf_p->get_ptr<T>());
        return;
    }
    // Chunked single-launch path: issue the WHOLE flat range as ONE
    // chunk-aware kernel instead of the per-piece driver's one launch per
    // chunk boundary crossing. The per-element arithmetic is byte-for-byte
    // the classic kernel's, so the result is bitwise the whole-buffer call.
    //
    // Transport: the device chunk-pointer ring is the only regime.
    {
        // Piece C (2026-10-18c): the all-1-chunk degenerate launch is
        // the FIRST thing the chunked block does -- the chunk counts
        // are read straight off the buffers, before any ChunkView /
        // RingTables / per-operand work exists, so the degenerate host
        // path is as lean as the classic dispatch above (gemm path-F
        // precedent; launch pointer-identical, bitwise by construction,
        // ring untouched).
        if(args->num_elems > 0
            && buf_grad->chunk_count() == 1
            && buf_vel->chunk_count() == 1
            && buf_p->chunk_count() == 1)
        {
            kernel::sgd_step::cuda<T>(stream, args->num_iter,
                args->num_elems, args->momentum, args->lr,
                args->weight_decay, args->dampening, args->nesterov,
                buf_grad->get_ptr<T>(), buf_vel->get_ptr<T>(),
                buf_p->get_ptr<T>());
            return;
        }
        chunkunits::ChunkView<T> const cv_grad(buf_grad);
        chunkunits::ChunkView<T> const cv_vel(buf_vel);
        chunkunits::ChunkView<T> const cv_p(buf_p);
        if(args->num_elems > 0)
        {
            // ONE merged table (grad | velocity | param) in the reusable
            // thread-local staging, ONE ring upload per launch.
            chunkunits::RingTables &tabs = chunkunits::ring_tables();
            tabs.clear();
            std::size_t const o_grad = tabs.add(cv_grad);
            std::size_t const o_vel = tabs.add(cv_vel);
            std::size_t const o_p = tabs.add(cv_p);
            chunkunits::ChunkPtrRing &ring = chunkunits::chunk_ptr_ring();
            if(ring.begin(stream))
            {
                void const *ring_base = ring.upload_tables(tabs);
                if(ring_base != nullptr)
                {
                    T *const *g_tab = ring.table_view<T *>(ring_base, o_grad);
                    T *const *v_tab = ring.table_view<T *>(ring_base, o_vel);
                    T *const *p_tab = ring.table_view<T *>(ring_base, o_p);
                    kernel::sgd_step::chunked<T>(
                        stream,
                        args->num_iter,
                        args->num_elems,
                        args->momentum,
                        args->lr,
                        args->weight_decay,
                        args->dampening,
                        args->nesterov,
                        g_tab,
                        Index(cv_grad.nchunks),
                        v_tab,
                        Index(cv_vel.nchunks),
                        p_tab,
                        Index(cv_p.nchunks),
                        Index(cv_grad.chunk_elems),
                        Index(cv_vel.chunk_elems),
                        Index(cv_p.chunk_elems));
                    ring.end(stream);
                    return;
                }
                    // (rule 11, user directive 2026-10-19) There is no per-piece
                    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot ring
                    // serve any launch up to a 16 TiB operand, so a failed upload is an
                    // invariant violation, not a mode to stage around.
                    std::fprintf(stderr,
                        "nntile_sgd_step: chunk-ptr ring cannot serve this launch (invariant "
                        "violation; the per-piece staging fallback was deleted by rule "
                        "11)\n");
                    std::abort();
            }
        }
    }
    if(args->num_elems > 0)
    {
        chunkunits::classic_dispatch_tripwire("nntile_sgd_step",
            false);
    }
    // Launch kernel per flat chunk piece (single piece if unchunked)
    auto grad = ::nntile::haul::buf_chunks<T>(buffers, 0);
    auto velocity = ::nntile::haul::buf_chunks<T>(buffers, 1);
    auto p = ::nntile::haul::buf_chunks<T>(buffers, 2);
    ::nntile::haul::for_each_flat_piece(
        args->num_elems,
        [&](std::size_t n, const T *g, T *v, T *pp) {
            kernel::sgd_step::cuda<T>(
                stream,
                args->num_iter,
                n,
                args->momentum,
                args->lr,
                args->weight_decay,
                args->dampening,
                args->nesterov,
                g,
                v,
                pp
            );
        },
        grad,
        velocity,
        p);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void SGDStep<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SGDStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SGDStep<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SGDStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SGDStep<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SGDStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for sgd_step tasks that depends only on cl_arg
template<typename T>
std::uint64_t SGDStep<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    bool first_iter = args->num_iter == 1;
    hash = ::nntile::haul::fnv1a(&first_iter, sizeof(bool), hash);
    hash = ::nntile::haul::fnv1a(&args->num_elems, sizeof(args->num_elems), hash);
    hash = ::nntile::haul::fnv1a(&args->momentum, sizeof(args->momentum), hash);
    hash = ::nntile::haul::fnv1a(&args->weight_decay, sizeof(args->weight_decay), hash);
    hash = ::nntile::haul::fnv1a(&args->dampening, sizeof(args->dampening), hash);
    hash = ::nntile::haul::fnv1a(&args->nesterov, sizeof(args->nesterov), hash);
    return hash;
}

//! Submit SGD step task
template<typename T>
void SGDStep<std::tuple<T>>::submit(int starpu_worker_hint,
    Index num_iter,
    Index num_elems,
    Scalar momentum,
    Scalar lr,
    Scalar weight_decay,
    Scalar dampening,
    bool nesterov,
    ::nnhaul::Handle & grad,
    ::nnhaul::Handle & velocity,
    ::nnhaul::Handle & param
)
{
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->num_iter = num_iter;
    args->num_elems = num_elems;
    args->momentum = momentum;
    args->lr = lr;
    args->weight_decay = weight_decay;
    args->dampening = dampening;
    args->nesterov = nesterov;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &grad }, { STARPU_RW, &velocity }, { STARPU_RW, &param } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SGDStep<std::tuple<nntile::fp64_t>>;
template class SGDStep<std::tuple<nntile::fp32_t>>;
template class SGDStep<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SGDStep<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SGDStep<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SGDStep<std::tuple<nntile::bf16_t>>;
template class SGDStep<std::tuple<nntile::fp16_t>>;

//! Pack of sgd_step operations for different types
sgd_step_pack_t sgd_step;

} // namespace nntile::haul
