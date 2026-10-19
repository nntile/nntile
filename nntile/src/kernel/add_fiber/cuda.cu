/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_fiber/cuda.cu
 * Per-element addition of a tensor and a broadcasted fiber on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/add_fiber/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_fiber
{

template<typename T>
static __global__
void cuda_kernel(
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha_,
    const T *src1,
    Scalar beta_,
    const T *src2,
    T *dst
)
//! Generic implementation of the add_fiber operation on CUDA
/*! @copydoc nntile::kernel::add_fiber::cuda
 * */
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x,
          i0 = threadIdx.y + blockIdx.y*blockDim.y,
          i1 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    if(i2 < k and i1 < n and i0 < m)
    {
        for(Index b = 0; b < batch; ++b)
        {
            // Value to add to the output slice
            const Y src1_val = alpha * static_cast<Y>(src1[i2+b*k]);
	        const T *src2_fiber = src2 + ((i1+b*n)*k+i2)*m;
            // Output fiber to be updated
            T *dst_fiber = dst + ((i1+b*n)*k+i2)*m;
            // Read value from the output
            const Y src2_val = static_cast<Y>(src2_fiber[i0]);
            // And update output
            dst_fiber[i0] = static_cast<T>(beta * src2_val + src1_val);
        }
    }
}

template<typename T>
void cuda(
    cudaStream_t stream,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    const T *src1,
    Scalar beta,
    const T *src2,
    T *dst
) noexcept
//! Add a tensor and a broadcasted fiber with optional scaling on CUDA
/*! Performs the following operation:
 * dst[i,l,j,b] = beta*src2[i,l,j,b] + alpha*src1[l,b]
 *
 * This function reads both src1 and src2 even if alpha or beta is zero.
 * If alpha is zero and src1[l,b] is NaN, then dst[i,l,j,b] will be NaN.
 * If beta is zero and src2[i,l,j,b] is NaN, then dst[i,l,j,b] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale, and in a case of beta being zero,
 * use nntile::kernel::scale_fiber instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale
 * @see nntile::kernel::scale_fiber
 * @see nntile::kernel::clear
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of dst tensor.
 * @param[in] n: Size of the last mode of dst tensor.
 * @param[in] k: Size of the middle mode of dst and src2 tensors, and the first
 * mode of src1.
 * @param[in] batch: Size of the batch dimension.
 * @param[in] alpha: Scalar factor for src1.
 * @param[in] src1: Input contiguous vector with k*batch elements.
 * @param[in] beta: Scaling factor for src2.
 * @param[in] src2: Input contiguous tensor with m*k*n*batch elements.
 * @param[inout] dst: Output contiguous m-by-k-by-n-by-batch tensor.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(k), 1024), std::min(int(m), 1),
            std::min(int(n), 1));
    dim3 blocks((k+threads.x-1)/threads.x, (m+threads.y-1)/threads.y,
            (n+threads.z-1)/threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, batch, alpha,
            src1, beta, src2, dst);
}

//! Chunk descriptors for the one-launch chunked add_fiber (see cuda.hh).
template<typename T>
struct AddFiberChunkTables
{
    const T* const* src1; //!< device-ring view of the src1 pointer array
    Index nsrc1;
    const T* const* src2; //!< device-ring view of the src2 pointer array
    Index nsrc2;
    T* const* dst; //!< device-ring view of the dst pointer array
    Index ndst;
    Index elems_src1; //!< elements in a full src1 chunk
    Index elems_src2; //!< elements in a full src2 chunk
    Index elems_dst; //!< elements in a full dst chunk
    int shift_src1; //!< log2(elems_src1) on a power of two, else -1
    int shift_src2; //!< log2(elems_src2) on a power of two, else -1
    int shift_dst; //!< log2(elems_dst) on a power of two, else -1
    Index m;
    Index n;
    Index k;
    Index batch;
    long total; //!< m*n*k*batch, the flat element-space size
    Scalar alpha;
    Scalar beta;
};

//! Chunk index of a flat element offset (shift on a power-of-two elems).
static __device__ __forceinline__ Index adfb_chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

//! Correctly-rounded FMA, to pin the association nvcc chose for cuda().
static __device__ __forceinline__ float adfb_fma(float a, float b, float c)
{
    return __fmaf_rn(a, b, c);
}
static __device__ __forceinline__ double adfb_fma(double a, double b,
    double c)
{
    return __fma_rn(a, b, c);
}

//! In-chunk element offset of flat offset f (mask when elems is pow2).
static __device__ __forceinline__ long adfb_off(long f, Index elems, int shift,
    Index c)
{
    return f & ((1LL << shift) - 1);
}

//! Variant of chunked_cuda_kernel without the per-block shared-memory
//! staging: src2/dst chunk pointers are read straight from the read-only
//! device ring (L2-resident, warp-uniform, so the load broadcasts) and
//! the src1 table is read directly.  Removes one __syncthreads and one
//! shared-memory fill per block.  Same per-element expression and same
//! result as the staged form (both go through the ring view), so it is
//! bitwise cuda()'s.
template<typename T>
static __global__
void chunked_cuda_kernel(AddFiberChunkTables<T> t)
{
    long const stride = (long)gridDim.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha = t.alpha;
    const Y beta = t.beta;
    for(long f = (long)blockIdx.x*blockDim.x + threadIdx.x;
            f < t.total; f += stride)
    {
        // Flat element index f = i0 + m*(i2 + k*(i1 + n*b)); resolve the
        // chunk of each operand from f (src1 flat is i2 + k*b).  The
        // GPT-2 shapes are m == 1 (the fiber is the contiguous mode), so
        // specialise that away from the divisions-by-m/n; the branch is
        // uniform across the block.
        Index i2;
        Index b;
        if(t.m == 1)
        {
            long const r2 = f / long(t.k);
            i2 = Index(f - r2*long(t.k));
            b = Index(t.n == 1 ? r2 : r2 / long(t.n));
        }
        else
        {
            long const r = f / long(t.m);
            i2 = Index(r % long(t.k));
            long const r2 = r / long(t.k);
            b = Index(r2 / long(t.n));
        }
        long const f1 = long(i2) + long(b)*long(t.k);
        Y s1;
        if(t.nsrc1 == 1)
        {
            s1 = Y{t.src1[0][f1]};
        }
        else
        {
            Index const c1 = adfb_chunk_of(f1, t.elems_src1, t.shift_src1);
            s1 = Y{t.src1[c1][adfb_off(f1, t.elems_src1, t.shift_src1, c1)]};
        }
        Index const c2 = adfb_chunk_of(f, t.elems_src2, t.shift_src2);
        const Y s2 = Y{t.src2[c2][adfb_off(f, t.elems_src2,
            t.shift_src2, c2)]};
        // Same association as cuda_kernel: fl(beta*src2), then
        // fma(alpha, src1, .).
        Y pr = beta * s2;
        Y const out = adfb_fma(alpha, s1, pr);
        // src2 and dst share a chunking in every driver call site; when
        // their chunk_elems match, reuse c2's chunk index and offset.
        if(t.elems_src2 == t.elems_dst && t.shift_src2 == t.shift_dst)
        {
            t.dst[c2][adfb_off(f, t.elems_dst, t.shift_dst, c2)] = T{out};
        }
        else
        {
            Index const cd = adfb_chunk_of(f, t.elems_dst, t.shift_dst);
            t.dst[cd][adfb_off(f, t.elems_dst, t.shift_dst, cd)] =
                T{out};
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T* const* src1_chunks, Index nsrc1, Scalar beta,
        const T* const* src2_chunks, Index nsrc2, T* const* dst_chunks,
        Index ndst, Index elems_src1, Index elems_src2, Index elems_dst)
    noexcept
{
    if(m <= 0 || n <= 0 || k <= 0 || batch <= 0 || nsrc1 <= 0 || nsrc2 <= 0
        || ndst <= 0)
    {
        return;
    }
    AddFiberChunkTables<T> t;
    t.src1 = src1_chunks;
    t.nsrc1 = nsrc1;
    t.src2 = src2_chunks;
    t.nsrc2 = nsrc2;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_src1 = elems_src1;
    t.elems_src2 = elems_src2;
    t.elems_dst = elems_dst;
    t.shift_src1 = chunk_shift_of<T>(t.nsrc1, elems_src1, "nntile_add_fiber");
    t.shift_src2 = chunk_shift_of<T>(t.nsrc2, elems_src2, "nntile_add_fiber");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_add_fiber");
    t.m = m;
    t.n = n;
    t.k = k;
    t.batch = batch;
    t.total = long(m)*long(n)*long(k)*long(batch);
    t.alpha = alpha;
    t.beta = beta;
    // Flat grid-stride over the whole element space: coalesced access and
    // far fewer blocks than the per-(i0,i1,i2) geometry, which measured
    // ~1.7x chunked vs ~1.04x here on the GPT-2 add_fiber shape.
    int const threads = 256;
    long const want = (t.total + threads - 1) / threads;
    int const blocks = want > 2048 ? 2048 : int(want > 0 ? want : 1);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp32_t *src1, Scalar beta,
	const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp64_t *src1, Scalar beta,
	const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const bf16_t *src1, Scalar beta,
	const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp16_t *src1, Scalar beta,
	const fp16_t *src2, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp32_t* const* src1_chunks,
        Index nsrc1, Scalar beta, const fp32_t* const* src2_chunks,
        Index nsrc2, fp32_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp64_t* const* src1_chunks,
        Index nsrc1, Scalar beta, const fp64_t* const* src2_chunks,
        Index nsrc2, fp64_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const bf16_t* const* src1_chunks,
        Index nsrc1, Scalar beta, const bf16_t* const* src2_chunks,
        Index nsrc2, bf16_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp16_t* const* src1_chunks,
        Index nsrc1, Scalar beta, const fp16_t* const* src2_chunks,
        Index nsrc2, fp16_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;






} // namespace nntile::kernel::add_fiber
