/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/fill/cuda.cu
 * Fill operation on CUDA
 *
 * @version 1.3.0
 * */

#include "nntile/kernel/fill/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::fill
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar val_, T *data)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y val{val_};
    if(i < nelems)
    {
        data[i] = T{val};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar val, T *data)
    noexcept
//! Fill operation on CUDA
/*! Sets all elements to the provided value
 * @params[in] nelems: Number of elements in a buffer
 * @param[in] val: Input value
 * @params[out] data: Output buffer
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, val, data);
}

//! Chunk descriptors for the one-launch chunked fill (see cuda.hh).
template<typename T>
struct ChunkTables
{
    T *const* data; //!< device-ring view of the data pointer array
    Index ndata;
    Index elems_data; //!< elements in a full data chunk
    int shift_data; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
};

//! Streaming store (STG.CS, evict-first) for the bulk fill path: the
//! filled bytes are identical, but a fill larger than the L2 marks its
//! lines evict-first so the write flood does not evict the operand's own
//! chunk-table line -- tail-wave blocks then still hit it in L2 instead
//! of re-fetching it from DRAM at the kernel end (a stable p90 tail on
//! the 6 MiB wpe fill). Only the fp32 storage instantiation takes the
//! intrinsic; every other element type keeps the plain store.
template<typename T>
static __device__ __forceinline__ void fill_store(T *p, T v)
{
    p[0] = v;
}

static __device__ __forceinline__ void fill_store(fp32_t *p, fp32_t v)
{
    __stcs(reinterpret_cast<float *>(p), v.value);
}

//! Chunk-aware fill: the SAME flat thread mapping (i = tid + bid*256) and
//! the SAME per-element store as cuda_kernel, with the destination address
//! resolved through its chunk table read straight from the device ring
//! (piece 9 / rule 10: no staging, no table __syncthreads). The result
//! is bitwise cuda()'s.
//!
//! A block whose 256-element window lies inside one chunk (the common
//! case: full chunks hold 2^20 elements, a multiple of the block size)
//! resolves its chunk base ONCE per block -- the sgd_step pattern -- so
//! the per-element address work is the classic kernel's own one IMAD and
//! the ring table is read one pointer per block, not per element.
//! Straddling windows (only possible inside the short tail chunk of a
//! non-full allocation) resolve every element through the table directly.
template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, Scalar val_, ChunkTables<T> t)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y val{val_};
    if(i >= nelems)
    {
        return;
    }
    long const w0 = long(blockIdx.x)*long(blockDim.x);
    long const off = w0 & ((1LL << t.shift_data) - 1);
    if(off + long(blockDim.x) <= long(t.elems_data))
    {
        // Whole block window inside chunk w0 >> shift: one base pointer,
        // then the classic kernel's per-element store (streaming, see
        // fill_store).
        T *p = t.data[w0 >> t.shift_data] + off + threadIdx.x;
        fill_store(p, T{val});
    }
    else
    {
        long const f = long(i);
        long const c = f >> t.shift_data;
        t.data[c][f - c * long(t.elems_data)] = T{val};
    }
}

//! Single-chunk fill (both the whole range and its own table's chunk 0):
//! the classic kernel's statement stream with the base pointer resolved
//! through ONE ring-table load issued as the very first statement (via
//! the read-only data path: the table is never written by this kernel),
//! so the launch floor pays exactly one dependent load and nothing else.
//! The fill value arrives pre-converted to the representation type
//! (exactly the conversion the classic kernel performs in-kernel, done
//! host-side once -- the sgd_step pattern), so the kernel body is guard +
//! store only. Threads past nelems issue the same broadcast table load
//! but no store; the stored bytes are bitwise cuda()'s.
template<typename T>
static __global__
void chunked_single_kernel(Index nelems, typename T::repr_t val_,
    T* const* data_tab)
{
    T *data = reinterpret_cast<T *>(__ldg(reinterpret_cast<
        const unsigned long long *>(data_tab)));
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        data[i] = T{val_};
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar val,
        T* const* data_chunks, Index ndata, Index elems_data)
    noexcept
{
    if(nelems <= 0 || ndata <= 0)
    {
        return;
    }
    if(ndata == 1)
    {
        // Dispatch by chunk count only. Same 256-thread geometry as the
        // whole-buffer entry.
        dim3 blocks((nelems+255)/256), threads(256);
        (chunked_single_kernel<T>)<<<blocks, threads, 0, stream>>>(
            nelems, typename T::repr_t(val), data_chunks);
        return;
    }
    ChunkTables<T> t;
    t.data = data_chunks;
    t.ndata = ndata;
    t.elems_data = elems_data;
    t.shift_data = chunk_shift_of<T>(t.ndata, elems_data, "nntile_fill");
    dim3 blocks((nelems+255)/256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(
        nelems, val, t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar val, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar val, fp64_t *data)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar val, bf16_t *data)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar val, fp16_t *data)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems, Scalar val,
        fp32_t* const* data_chunks, Index ndata, Index elems_data) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems, Scalar val,
        fp64_t* const* data_chunks, Index ndata, Index elems_data) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems, Scalar val,
        bf16_t* const* data_chunks, Index ndata, Index elems_data) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems, Scalar val,
        fp16_t* const* data_chunks, Index ndata, Index elems_data) noexcept;

} // namespace nntile::kernel::fill
