/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sgd_step/cuda.cu
 * SGD with momentum step with buffers on CUDA
 *
 * @version 1.2.0
 * */

#include "nntile/kernel/sgd_step/cuda.hh"
#include "nntile/kernel/cuda.hh"

#include <cstdio>

namespace nntile::kernel::sgd_step
{

template<typename T>
static __global__
void cuda_kernel(Index num_iter, Index num_elems, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
        const T *grad, T *velocity, T *p)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    if(i < num_elems)
    {
        // Read values (param+grad) from RAM only once
        Y p_val = static_cast<Y>(p[i]), grad_val = static_cast<Y>(grad[i]);
        if (weight_decay != 0)
        {
            grad_val += weight_decay * p_val;
        }
        if (momentum != 0.0)
        {
            Y velocity_val;
            if (num_iter == 1)
            {
                velocity_val = grad_val;
            }
            else
            {
                velocity_val = momentum * static_cast<Y>(velocity[i]) +
                    static_cast<Y>(1.0 - dampening) * grad_val;
            }
            velocity[i] = static_cast<T>(velocity_val);
            if (nesterov)
            {
                grad_val += momentum * velocity_val;
            }
            else
            {
                grad_val = velocity_val;
            }
        }
        p[i] = static_cast<T>(p_val - lr * grad_val);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index num_iter, Index num_elems, Scalar momentum, Scalar lr,
        Scalar weight_decay, Scalar dampening, bool nesterov, const T *grad_, T *velocity_, T *p_)
    noexcept
//! Fused SGD with momentum step operation of buffers
/*!
* @param[in] stream: CUDA stream
* @param[in] num_iter: current iteration number
* @param[in] num_elems: Number of elements in buffers
* @param[in] momentum: momentum coefficient
* @param[in] lr: learning rate
* @param[in] weight_decay: coefficient for l2 regularizer
* @param[in] dampening: dampening coefficient for momentum
* @param[in] nesterov: whether to use Nesterov momentum
* @param[in] grad: Input buffer stored gradient
* @param[inout] velocity: Input buffer stored velocity (momentum buffer)
* @param[inout] p: Input buffers with parameter that are updated in the end
 * */
{
    dim3 blocks((num_elems+255)/256), threads(256);
    using Y = typename T::repr_t;
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(num_iter, num_elems,
            Y{momentum}, Y{lr}, Y{weight_decay}, Y{dampening}, nesterov, grad_, velocity_, p_);
}

//! Chunk descriptors for the one-launch chunked sgd_step (cuda.hh).
template<typename T>
struct SgdStepChunkTables
{
    const T *const *grad; //!< device view of the grad pointer array
    Index ngrad;
    T *const *vel; //!< device view of the velocity pointer array
    Index nvel;
    T *const *p; //!< device view of the parameter pointer array
    Index np;
    Index elems_grad; //!< elements in a full grad chunk
    Index elems_vel; //!< elements in a full velocity chunk
    Index elems_p; //!< elements in a full parameter chunk
    int shift_grad; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_vel; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_p; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    Index num_iter;
    typename T::repr_t momentum;
    typename T::repr_t lr;
    typename T::repr_t weight_decay;
    typename T::repr_t dampening;
    bool nesterov;
    Index num_elems;
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ long sgs_chunk_of(long f, Index elems,
    int shift) noexcept
{
    return f >> shift;
}

//! Explicitly-rounded FP helpers (one rounding per operation, no FMA
//! contraction freedom): nvcc's default contraction can differ between
//! the chunked kernel and the classic kernel (observed: one fused the
//! momentum update, the other did not), which broke the bitwise-vs-
//! classic guarantee for exactly one ulp. Routing every FP operation of
//! the update through these intrinsics makes the chunked kernel -- and
//! the classic kernel's as-compiled unfused arithmetic -- agree by
//! construction. (Verified by sgd_step_chunked_check.cu over all
//! momentum/weight-decay/dampening/nesterov branches.)
static __device__ __forceinline__ float sgs_mul(float a, float b) noexcept
{
    return __fmul_rn(a, b);
}
static __device__ __forceinline__ double sgs_mul(double a, double b) noexcept
{
    return __dmul_rn(a, b);
}
static __device__ __forceinline__ float sgs_add(float a, float b) noexcept
{
    return __fadd_rn(a, b);
}
static __device__ __forceinline__ double sgs_add(double a, double b) noexcept
{
    return __dadd_rn(a, b);
}
static __device__ __forceinline__ float sgs_sub(float a, float b) noexcept
{
    return __fsub_rn(a, b);
}
static __device__ __forceinline__ double sgs_sub(double a, double b) noexcept
{
    return __dsub_rn(a, b);
}
static __device__ __forceinline__ float sgs_fma(float a, float b, float c)
    noexcept
{
    return __fmaf_rn(a, b, c);
}
static __device__ __forceinline__ double sgs_fma(double a, double b, double c)
    noexcept
{
    return __fma_rn(a, b, c);
}

//! In-chunk offset of a flat element offset (mask for a power of two).
static __device__ __forceinline__ long sgs_off_of(long f, Index elems,
    int shift) noexcept
{
    return f & ((1LL << shift) - 1);
}

//! Chunk-aware sgd_step: the classic kernel's geometry (one element per
//! thread, 256 threads per block) with the classic kernel's per-element
//! arithmetic verbatim, so the whole-range result is bitwise one cuda()
//! call. A block whose 256-element window lies inside a single chunk of
//! every operand (the common case: full chunks hold 2^20 elements, a
//! multiple of the block size) reads just its own three chunk base
//! pointers straight from the ring -- no shared-memory table staging, no
//! __syncthreads, so the per-block cost over the classic kernel is three
//! 8-byte loads. Straddling blocks (non-power-of-two chunk element
//! counts) resolve every element through the ring tables directly
//! (piece 9 / rule 10: no staging, no table __syncthreads).
template<typename T, typename Tab>
static __global__
void chunked_cuda_kernel(Tab t)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i >= t.num_elems)
    {
        return;
    }
    long const f = long(i);
    long const w0 = long(blockIdx.x)*long(blockDim.x);
    long const wlen = long(blockDim.x);
    long const og = sgs_off_of(w0, t.elems_grad, t.shift_grad);
    long const ov = sgs_off_of(w0, t.elems_vel, t.shift_vel);
    long const op = sgs_off_of(w0, t.elems_p, t.shift_p);
    bool const one_g = (og + wlen) <= long(t.elems_grad);
    bool const one_v = (ov + wlen) <= long(t.elems_vel);
    bool const one_p = (op + wlen) <= long(t.elems_p);
    using Y = typename T::repr_t;
    if(one_g and one_v and one_p)
    {
        // Plain per-block base pointers, byte-for-byte the classic kernel's
        // code (no table addressing left in this branch).
        const T *g = t.grad[sgs_chunk_of(w0, t.elems_grad, t.shift_grad)]
            + og + threadIdx.x;
        T *v = t.vel[sgs_chunk_of(w0, t.elems_vel, t.shift_vel)]
            + ov + threadIdx.x;
        T *pp = t.p[sgs_chunk_of(w0, t.elems_p, t.shift_p)]
            + op + threadIdx.x;
        // Read values (param+grad) from RAM only once
        Y p_val = static_cast<Y>(pp[0]);
        Y grad_val = static_cast<Y>(g[0]);
        if (t.weight_decay != 0)
        {
            grad_val += t.weight_decay * p_val;
        }
        if (t.momentum != 0.0)
        {
            Y velocity_val;
            if (t.num_iter == 1)
            {
                velocity_val = grad_val;
            }
            else
            {
                // The momentum update is pinned to the classic kernel's
                // exact as-compiled rounding: nvcc's default FMA
                // contraction differs between the chunked kernel and the
                // classic kernel, which broke bitwise equality with the
                // classic kernel by one ulp. Empirically (12345-element
                // probe, all elements matched) the classic kernel computes
                //   fma((Y)(1-dampening), grad, mom*v)
                // with mom*v rounded separately and the dampening product
                // fused into the add; reproduce that exactly.
                Y const damp_term = static_cast<Y>(1.0 - t.dampening);
                velocity_val = sgs_fma(damp_term, grad_val,
                    sgs_mul(t.momentum, static_cast<Y>(v[0])));
            }
            v[0] = static_cast<T>(velocity_val);
            if (t.nesterov)
            {
                grad_val += t.momentum * velocity_val;
            }
            else
            {
                grad_val = velocity_val;
            }
        }
        pp[0] = static_cast<T>(p_val - t.lr * grad_val);
    }
    else
    {
        // Window straddles a chunk boundary of some operand: resolve every
        // element through the ring tables directly (rule 10: no staging,
        // no table __syncthreads; same arithmetic).
        Y p_val = static_cast<Y>(
            t.p[sgs_chunk_of(f, t.elems_p, t.shift_p)][sgs_off_of(f, t.elems_p, t.shift_p)]);
        Y grad_val = static_cast<Y>(
            t.grad[sgs_chunk_of(f, t.elems_grad, t.shift_grad)][sgs_off_of(f, t.elems_grad, t.shift_grad)]);
        if (t.weight_decay != 0)
        {
            grad_val += t.weight_decay * p_val;
        }
        if (t.momentum != 0.0)
        {
            Y velocity_val;
            if (t.num_iter == 1)
            {
                velocity_val = grad_val;
            }
            else
            {
                // Same exact-rounding pin as the fast branch (see the
                // comment there).
                Y const damp_term = static_cast<Y>(1.0 - t.dampening);
                velocity_val = sgs_fma(damp_term, grad_val,
                    sgs_mul(t.momentum, static_cast<Y>(
                    t.vel[sgs_chunk_of(f, t.elems_vel, t.shift_vel)][sgs_off_of(f, t.elems_vel, t.shift_vel)])));
            }
            t.vel[sgs_chunk_of(f, t.elems_vel, t.shift_vel)]
                [sgs_off_of(f, t.elems_vel, t.shift_vel)]
                = static_cast<T>(velocity_val);
            if (t.nesterov)
            {
                grad_val += t.momentum * velocity_val;
            }
            else
            {
                grad_val = velocity_val;
            }
        }
        t.p[sgs_chunk_of(f, t.elems_p, t.shift_p)]
            [sgs_off_of(f, t.elems_p, t.shift_p)]
            = static_cast<T>(p_val - t.lr * grad_val);
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening,
        bool nesterov, const T *const *grad_chunks, Index ngrad,
        T *const *vel_chunks, Index nvel, T *const *p_chunks, Index np,
        Index elems_grad, Index elems_vel, Index elems_p)
    noexcept
{
    if(num_elems <= 0 || ngrad <= 0 || nvel <= 0 || np <= 0)
    {
        return;
    }
    SgdStepChunkTables<T> t;
    t.grad = grad_chunks;
    t.ngrad = ngrad;
    t.vel = vel_chunks;
    t.nvel = nvel;
    t.p = p_chunks;
    t.np = np;
    t.elems_grad = elems_grad > 0 ? elems_grad : 1;
    t.elems_vel = elems_vel > 0 ? elems_vel : 1;
    t.elems_p = elems_p > 0 ? elems_p : 1;
    t.shift_grad = chunk_shift_of<T>(t.ngrad, t.elems_grad, "nntile_sgd_step");
    t.shift_vel = chunk_shift_of<T>(t.nvel, t.elems_vel, "nntile_sgd_step");
    t.shift_p = chunk_shift_of<T>(t.np, t.elems_p, "nntile_sgd_step");
    t.num_iter = num_iter;
    using Y = typename T::repr_t;
    t.momentum = Y{momentum};
    t.lr = Y{lr};
    t.weight_decay = Y{weight_decay};
    t.dampening = Y{dampening};
    t.nesterov = nesterov;
    t.num_elems = num_elems;
    dim3 blocks((num_elems+255)/256), threads(256);
    (chunked_cuda_kernel<T, SgdStepChunkTables<T>>)<<<blocks, threads,
        0, stream>>>(t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index num_iter, Index num_elems, Scalar momentum,
        Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov, const fp32_t *grad, fp32_t *velocity,
        fp32_t *p)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index num_iter, Index num_elems, Scalar momentum,
        Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov, const fp64_t *grad, fp64_t *velocity,
        fp64_t *p)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index num_iter, Index num_elems, Scalar momentum,
        Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov, const bf16_t *grad, bf16_t *velocity,
        bf16_t *p)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index num_iter, Index num_elems, Scalar momentum,
        Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov, const fp16_t *grad, fp16_t *velocity,
        fp16_t *p)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening,
        bool nesterov, const fp32_t *const *grad_chunks, Index ngrad,
        fp32_t *const *vel_chunks, Index nvel, fp32_t *const *p_chunks,
        Index np, Index elems_grad, Index elems_vel, Index elems_p)
    noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening,
        bool nesterov, const fp64_t *const *grad_chunks, Index ngrad,
        fp64_t *const *vel_chunks, Index nvel, fp64_t *const *p_chunks,
        Index np, Index elems_grad, Index elems_vel, Index elems_p)
    noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening,
        bool nesterov, const bf16_t *const *grad_chunks, Index ngrad,
        bf16_t *const *vel_chunks, Index nvel, bf16_t *const *p_chunks,
        Index np, Index elems_grad, Index elems_vel, Index elems_p)
    noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening,
        bool nesterov, const fp16_t *const *grad_chunks, Index ngrad,
        fp16_t *const *vel_chunks, Index nvel, fp16_t *const *p_chunks,
        Index np, Index elems_grad, Index elems_vel, Index elems_p)
    noexcept;

} // namespace nntile::kernel::sgd_step
