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

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ long fill_chunk_index(long f, Index elems,
    int shift) noexcept
{
    return f >> shift;
}

//! Chunk-aware fill: the SAME flat thread mapping (i = tid + bid*256) and
//! the SAME per-element store as cuda_kernel, with the destination address
//! resolved through its chunk table read straight from the device ring
//! (piece 9 / rule 10: no staging, no table __syncthreads). The result
//! is bitwise cuda()'s.
template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, Scalar val_, ChunkTables<T> t)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y val{val_};
    if(i < nelems)
    {
        long const f = long(i);
        Index const c = Index(fill_chunk_index(f, t.elems_data, t.shift_data));
        t.data[c][f - c * long(t.elems_data)] = T{val};
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
