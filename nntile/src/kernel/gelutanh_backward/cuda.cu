/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelutanh_backward/cuda.cu
 * Backward approximate GeLU operation on CUDA
 *
 * @version 1.3.0
 * */

#include "nntile/kernel/gelutanh_backward/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::gelutanh_backward
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha, const T *x, const T *dy, Scalar beta, T *dx)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    // Constants
    using Y = typename T::repr_t;
    const Y alpha_{alpha}, beta_{beta};
    constexpr Y pi = 3.141592653589793238462643383279502884L,
        one = 1, f1 = Y{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const Y sqrt_pi = sqrt(pi), sqrt_2 = sqrt(Y{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2}*f2, f4 = f3*f1, f5 = Y{3}*f4;
    if(i < nelems)
    {
        Y z = Y{x[i]};
        Y z2 = z * z;
        Y y1 = z * (f3 + f4*z2);
        Y y2 = z * (f3 + f5*z2);
        Y expy1 = exp(y1);
        if(not isinf(expy1))
        {
            Y inv_expy1p1 = one / (expy1 + one);
            Y g = (one-y2*(one-inv_expy1p1)) * inv_expy1p1;
            Y contrib = alpha_ * g * Y{dy[i]};
            if(beta == 0.0)
            {
                dx[i] = T{contrib};
            }
            else
            {
                dx[i] = T{contrib + beta_ * Y{dx[i]}};
            }
        }
        else
        {
            if(beta == 0.0)
            {
                dx[i] = T{Y{0.0}};
            }
            else if(beta != 1.0)
            {
                dx[i] = T{beta_ * Y{dx[i]}};
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *x, const T *dy,
        Scalar beta, T *dx)
    noexcept
//! Backward approximate GeLU operation on CUDA
/*! Does the following per-element operation:
 * dx[i] = alpha * dy[i]*GeLUtanh'(x[i]) + beta * dx[i]
 * GeLUtanh'(z) = (1-(zf'(z)-1)exp(f(z))) / (1+exp(f(z)))^2
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] alpha: Scalar multiplier for the gradient contribution
 * @params[in] x: Input value for forward GeLU
 * @params[in] dy: Gradient over output of forward GeLU
 * @params[in] beta: Scalar multiplier for the existing dx value
 * @params[inout] dx: Gradient over input of forward GeLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, x, dy, beta, dx);
}

//! Chunk descriptors for the one-launch chunked gelutanh_backward (see
//! cuda.hh).
template<typename T>
struct ChunkTables
{
    const T *const* x; //!< device-ring view of the x pointer array
    Index nx;
    const T *const* dy; //!< device-ring view of the dy pointer array
    Index ndy;
    T *const* dx; //!< device-ring view of the dx pointer array
    Index ndx;
    Index elems_x; //!< elements in a full x chunk
    Index elems_dy; //!< elements in a full dy chunk
    Index elems_dx; //!< elements in a full dx chunk
    int shift_x; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_dy; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_dx; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ long gtb_chunk_index(long f, Index elems,
    int shift) noexcept
{
    return f >> shift;
}

//! Explicitly-rounded FP helpers (one rounding per operation, no FMA
//! contraction freedom): nvcc's default contraction can differ between
//! the chunked kernel and the classic kernel for the beta update
//! (contrib + beta*dx), which broke bitwise equality by one ulp -- the
//! same hazard sgd_step hit (see its sgs_* helpers). The helpers pin the
//! classic kernel's as-compiled rounding by construction.
static __device__ __forceinline__ float gtb_mul(float a, float b) noexcept
{
    return __fmul_rn(a, b);
}
static __device__ __forceinline__ double gtb_mul(double a, double b) noexcept
{
    return __dmul_rn(a, b);
}
static __device__ __forceinline__ float gtb_fma(float a, float b, float c)
    noexcept
{
    return __fmaf_rn(a, b, c);
}
static __device__ __forceinline__ double gtb_fma(double a, double b, double c)
    noexcept
{
    return __fma_rn(a, b, c);
}

//! Chunk-aware gelutanh_backward: the SAME flat thread mapping
//! (i = tid + bid*256) and the SAME per-element expression (copied
//! verbatim from cuda_kernel, including the per-thread constant
//! derivation and the beta special cases, so nvcc's contraction decisions
//! are the classic kernel's) with every operand address resolved through
//! its chunk table read straight from the device ring (piece 9 / rule
//! 10: no staging, no table __syncthreads). The result is bitwise
//! cuda()'s.
template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, Scalar alpha, Scalar beta,
        ChunkTables<T> t)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    // Constants
    using Y = typename T::repr_t;
    const Y alpha_{alpha}, beta_{beta};
    constexpr Y pi = 3.141592653589793238462643383279502884L,
        one = 1, f1 = Y{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const Y sqrt_pi = sqrt(pi), sqrt_2 = sqrt(Y{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2}*f2, f4 = f3*f1, f5 = Y{3}*f4;
    if(i < nelems)
    {
        long const f = long(i);
        Index const cx = Index(gtb_chunk_index(f, t.elems_x, t.shift_x));
        Index const cdy = Index(gtb_chunk_index(f, t.elems_dy, t.shift_dy));
        Index const cdx = Index(gtb_chunk_index(f, t.elems_dx, t.shift_dx));
        T *dxp = t.dx[cdx] + (f - cdx * long(t.elems_dx));
        Y z = Y{t.x[cx][f - cx * long(t.elems_x)]};
        Y z2 = z * z;
        Y y1 = z * (f3 + f4*z2);
        Y y2 = z * (f3 + f5*z2);
        Y expy1 = exp(y1);
        if(not isinf(expy1))
        {
            Y inv_expy1p1 = one / (expy1 + one);
            Y g = (one-y2*(one-inv_expy1p1)) * inv_expy1p1;
            Y contrib = alpha_ * g * Y{t.dy[cdy][f - cdy * long(t.elems_dy)]};
            if(beta == 0.0)
            {
                dxp[0] = T{contrib};
            }
            else
            {
                // Pinned to the classic kernel's as-compiled rounding:
                // fma(beta, dx, contrib) with contrib's own product
                // rounded separately.
                dxp[0] = T{gtb_fma(beta_, Y{dxp[0]}, contrib)};
            }
        }
        else
        {
            if(beta == 0.0)
            {
                dxp[0] = T{Y{0.0}};
            }
            else if(beta != 1.0)
            {
                dxp[0] = T{beta_ * Y{dxp[0]}};
            }
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        const T* const* x_chunks, Index nx, const T* const* dy_chunks,
        Index ndy, Scalar beta, T* const* dx_chunks, Index ndx,
        Index elems_x, Index elems_dy, Index elems_dx)
    noexcept
{
    if(nelems <= 0 || nx <= 0 || ndy <= 0 || ndx <= 0)
    {
        return;
    }
    ChunkTables<T> t;
    t.x = x_chunks;
    t.nx = nx;
    t.dy = dy_chunks;
    t.ndy = ndy;
    t.dx = dx_chunks;
    t.ndx = ndx;
    t.elems_x = elems_x;
    t.elems_dy = elems_dy;
    t.elems_dx = elems_dx;
    t.shift_x = chunk_shift_of<T>(t.nx, elems_x, "nntile_gelutanh_backward");
    t.shift_dy = chunk_shift_of<T>(t.ndy, elems_dy, "nntile_gelutanh_backward");
    t.shift_dx = chunk_shift_of<T>(t.ndx, elems_dx, "nntile_gelutanh_backward");
    dim3 blocks((nelems+255)/256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads,
        0, stream>>>(
        nelems, alpha, beta, t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp32_t *x,
        const fp32_t *dy, Scalar beta, fp32_t *dx)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp64_t *x,
        const fp64_t *dy, Scalar beta, fp64_t *dx)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const bf16_t *x,
        const bf16_t *dy, Scalar beta, bf16_t *dx)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp16_t *x,
        const fp16_t *dy, Scalar beta, fp16_t *dx)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t* const* x_chunks, Index nx, const fp32_t* const* dy_chunks,
        Index ndy, Scalar beta, fp32_t* const* dx_chunks, Index ndx,
        Index elems_x, Index elems_dy, Index elems_dx) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t* const* x_chunks, Index nx, const fp64_t* const* dy_chunks,
        Index ndy, Scalar beta, fp64_t* const* dx_chunks, Index ndx,
        Index elems_x, Index elems_dy, Index elems_dx) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t* const* x_chunks, Index nx, const bf16_t* const* dy_chunks,
        Index ndy, Scalar beta, bf16_t* const* dx_chunks, Index ndx,
        Index elems_x, Index elems_dy, Index elems_dx) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t* const* x_chunks, Index nx, const fp16_t* const* dy_chunks,
        Index ndy, Scalar beta, fp16_t* const* dx_chunks, Index ndx,
        Index elems_x, Index elems_dy, Index elems_dx) noexcept;

} // namespace nntile::kernel::gelutanh_backward
