/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/embedding_backward/cuda.cu
 * Backward of embeddings from vocabulary within buffers
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/embedding_backward/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::embedding_backward
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index k_start, Index k_size,
        Index index_range, Scalar alpha, const Index *index,
        const T *embed_, T *vocab)
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x;
    Index i0 = blockIdx.y, i1 = blockIdx.z;
    using Z = typename CUDAComputeType<T>::value;
    using Y = typename T::repr_t;
    if(i2 < k_size)
    {
        // The token becomes a raw pointer offset below: validate it
        // first and trap loudly (like a PyTorch device assert) instead
        // of corrupting memory.
        const long long token = static_cast<long long>(index[i1*m+i0]);
        if(token < 0 or token >= static_cast<long long>(index_range))
        {
            printf("nntile embedding_backward: token id %lld is outside "
                    "the vocabulary range [0, %lld)\n", token,
                    static_cast<long long>(index_range));
            __trap();
        }
        Z *vocab_slice = reinterpret_cast<Z*>(vocab + k_size*token);
        const Z *embed = reinterpret_cast<const Z *>(embed_);
        const Y contrib = static_cast<Y>(alpha) *
            static_cast<Y>(embed[(i1*k+k_start+i2)*m + i0]);
        atomicAdd(&vocab_slice[i2], static_cast<Z>(contrib));
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, Index vocab_nelems, Scalar alpha,
        Scalar beta, const int64_t *index_, const T *embed, T *vocab)
    noexcept
{
    if(beta == Scalar{0.0})
    {
        cudaMemsetAsync(vocab, 0, static_cast<size_t>(vocab_nelems) * sizeof(T),
                stream);
    }
    dim3 threads(256, 1, 1);
    dim3 blocks((k_size+255)/256, m, n);
    using I = typename CUDAComputeType<int64_t>::value;
    auto index = reinterpret_cast<const I *>(index_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, k_start,
            k_size, index_range,
            alpha, index, embed, vocab);
}

//! Chunk descriptors for the one-launch chunked embedding backward
//! (see cuda.hh).
template<typename T>
struct EmbBwdChunkTables
{
    const int64_t *const *index; //!< device-ring view of the index table
    const T *const *embed; //!< device-ring view of the embed table
    T *const *vocab; //!< device-ring view of the vocab table
    Index nindex;
    Index nembed;
    Index nvocab;
    Index elems_index; //!< elements in a full index chunk
    Index elems_embed; //!< elements in a full embed chunk
    Index elems_vocab; //!< elements in a full vocab chunk
    int shift_index; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_embed; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_vocab; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    Index m;
    Index n;
    Index k;
    Index k_start;
    Index k_size;
    Index index_range;
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ Index chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

//! In-chunk offset given the PRECOMPUTED in-chunk mask (hoisted once
//! per kernel; sentinel-62's mask is all-ones). Piece 7.
static __device__ __forceinline__ long chunk_off(long f, long mask)
{
    return f & mask;
}

//! Chunk-aware embedding backward scatter-add: the same (i2, i0, i1)
//! thread geometry and atomicAdd as cuda_kernel above, but each of the
//! three operands is addressed through its chunk table. Piece 9 (rule
//! 10): NO shared-memory staging -- every thread reads its chunk pointer
//! straight from the device ring (a handful of L1-resident entries), so
//! the kernel carries no table __syncthreads at all; the in-chunk masks
//! stay hoisted into registers once per kernel (piece 7).
template<typename T>
static __global__
void chunked_cuda_kernel(EmbBwdChunkTables<T> t, Scalar alpha_)
{
    using I = typename CUDAComputeType<int64_t>::value;
    using Z = typename CUDAComputeType<T>::value;
    using Y = typename T::repr_t;
    long const index_mask = (1LL << t.shift_index) - 1;
    long const embed_mask = (1LL << t.shift_embed) - 1;
    long const vocab_mask = (1LL << t.shift_vocab) - 1;
    Index const i2 = threadIdx.x + blockIdx.x*blockDim.x;
    Index const i0 = blockIdx.y;
    Index const i1 = blockIdx.z;
    if(i2 < t.k_size)
    {
        // Index element (flat i1*m + i0), chunk-resolved.
        long const fi = long(i1) * long(t.m) + long(i0);
        Index const ci = chunk_of(fi, t.elems_index, t.shift_index);
        I const *ip = reinterpret_cast<I const *>(t.index[ci])
            + chunk_off(fi, index_mask);
        const long long token = static_cast<long long>(ip[0]);
        if(token < 0 or token >= static_cast<long long>(t.index_range))
        {
            printf("nntile embedding_backward: token id %lld is outside "
                    "the vocabulary range [0, %lld)\n", token,
                    static_cast<long long>(t.index_range));
            __trap();
        }
        // Embed element (flat (i1*k + k_start + i2)*m + i0), chunk-resolved.
        long const fe = (long(i1) * long(t.k) + long(t.k_start) + long(i2))
            * long(t.m) + long(i0);
        Index const ce = chunk_of(fe, t.elems_embed, t.shift_embed);
        T const *ep = t.embed[ce] + chunk_off(fe, embed_mask);
        const Y contrib = static_cast<Y>(alpha_) * static_cast<Y>(ep[0]);
        // Vocabulary element (flat k_size*token + i2), chunk-resolved.
        long const fv = long(t.k_size) * long(token) + long(i2);
        Index const cv = chunk_of(fv, t.elems_vocab, t.shift_vocab);
        T *vbp = t.vocab[cv];
        Z *vp = reinterpret_cast<Z *>(vbp + chunk_off(fv, vocab_mask));
        atomicAdd(vp, static_cast<Z>(contrib));
    }
}

//! Zero every element of a chunked buffer in one launch. The fp32-class
//! path uses 16-byte stores when the chunk size is a power-of-two multiple
//! of four elements (the chunk-size ladder: chunk bases are >=256 B aligned
//! and every vector lies inside one chunk), matching cudaMemsetAsync's
//! store width; a scalar path covers every other type/geometry.
template<typename T>
static __global__
void chunked_zero_cuda_kernel(EmbBwdChunkTables<T> t, long total)
{
    using Z = typename CUDAComputeType<T>::value;
    long const tid = long(blockIdx.x) * long(blockDim.x) + long(threadIdx.x);
    long const nthreads = long(gridDim.x) * long(blockDim.x);
    if constexpr (sizeof(Z) == 4)
    {
        if(t.shift_vocab >= 2)
        {
            long const nvec = total >> 2;
            uint4 const z = make_uint4(0u, 0u, 0u, 0u);
            for(long v = tid; v < nvec; v += nthreads)
            {
                long const f = v << 2;
                Index const c = chunk_of(f, t.elems_vocab, t.shift_vocab);
                long const cbeg = long(c) * long(t.elems_vocab);
                long const off = f - cbeg;
                long const clen = total - cbeg < long(t.elems_vocab)
                    ? total - cbeg : long(t.elems_vocab);
                Z *p = reinterpret_cast<Z *>(t.vocab[c]) + off;
                if(off + 4 <= clen)
                {
                    // Streaming store: the bytes are fully overwritten, so
                    // bypass the L2 write-allocate (matches cudaMemsetAsync's
                    // store behaviour).
                    __stcs(reinterpret_cast<uint4 *>(p), z);
                }
                else
                {
                    long const n = clen - off;
                    for(long k = 0; k < n; ++k)
                    {
                        p[k] = static_cast<Z>(0);
                    }
                }
            }
            // Tail: total is not a multiple of four elements.
            for(long f = (nvec << 2) + tid; f < total; f += nthreads)
            {
                Index const c = chunk_of(f, t.elems_vocab, t.shift_vocab);
                reinterpret_cast<Z *>(t.vocab[c])[f - long(c)
                    * long(t.elems_vocab)] = static_cast<Z>(0);
            }
            return;
        }
    }
    for(long f = tid; f < total; f += nthreads)
    {
        Index const c = chunk_of(f, t.elems_vocab, t.shift_vocab);
        reinterpret_cast<Z *>(t.vocab[c])[f - long(c) * long(t.elems_vocab)]
            = static_cast<Z>(0);
    }
}

//! Strided-walk chunk-aware embedding backward (piece B, 2026-10-18c):
//! the classic geometry verbatim (one 256-thread block per (i0, i1)
//! occurrence, lanes at consecutive i2, so every atomicAdd instruction
//! targets one contiguous vocab run and every embed load keeps the
//! classic stride-m pattern -- coalescing and occupancy unchanged), but
//! each thread reads its token (and validates it) ONCE and then walks
//! the k axis in blockDim.x strides with REGISTER chunk pointers that
//! rebase from the ring only at full-chunk boundaries (the advance is a
//! plain increment otherwise, interlaced behind the issued atomic of the
//! same iteration; guarded so the final iteration never reads past the
//! table's last entry). Compile-time chunk shifts (the IDX1/EMB1/VOC1
//! template flags pick the 62 single-chunk sentinel -- no chunk
//! arithmetic at all -- or the rule-7 class shift) and 32-bit flat
//! arithmetic replace the runtime shifts and 64-bit masks of the
//! elementwise body, which stays as the huge-tile fallback (U32=false:
//! any operand's flat span >= 2^32; plus the 256*m embed-stride overflow
//! guard for extreme m with tiny k*n). Same atomicAdds to the same
//! addresses: within the classic kernel's atomic-ordering floor.
template<typename T, bool IDX1, bool EMB1, bool VOC1>
static __global__
void chunked_strided_cuda_kernel(EmbBwdChunkTables<T> t, Scalar alpha_)
{
    using I = typename CUDAComputeType<int64_t>::value;
    using Z = typename CUDAComputeType<T>::value;
    using Y = typename T::repr_t;
    constexpr std::size_t class_elems = kChunkClassBytes / sizeof(T);
    constexpr int IDX_SHIFT = IDX1 ? kChunkSingleShift
        : chunk_log2(kChunkClassBytes / sizeof(I));
    constexpr int VOC_SHIFT = VOC1 ? kChunkSingleShift
        : chunk_log2(class_elems);
    constexpr int EMB_SHIFT = EMB1 ? kChunkSingleShift
        : chunk_log2(class_elems);
    constexpr unsigned VOC_MASK = VOC1 ? 0xffffffffu
        : (1u << VOC_SHIFT) - 1u;
    constexpr unsigned EMB_MASK = EMB1 ? 0xffffffffu
        : (1u << EMB_SHIFT) - 1u;
    Index const i0 = blockIdx.y;
    Index const i1 = blockIdx.z;
    unsigned const u_m = unsigned(t.m);
    // One token per thread (validated exactly like the classic kernel).
    unsigned const u_fi = unsigned(i1)*u_m + unsigned(i0);
    I const *ip = IDX1
        ? reinterpret_cast<I const *>(t.index[0]) + u_fi
        : reinterpret_cast<I const *>(t.index[u_fi >> IDX_SHIFT])
            + (u_fi & ((1u << IDX_SHIFT) - 1u));
    const long long token = static_cast<long long>(ip[0]);
    if(token < 0 or token >= static_cast<long long>(t.index_range))
    {
        printf("nntile embedding_backward: token id %lld is outside "
                "the vocabulary range [0, %lld)\n", token,
                static_cast<long long>(t.index_range));
        __trap();
    }
    unsigned const row = unsigned(t.k_size)*unsigned(token);
    unsigned const fe_col = unsigned(i1)*unsigned(t.k) + unsigned(t.k_start);
    // Chunk-walk state: rebases from the ring only at full-chunk
    // boundaries (in-chunk offset wrapping below the stride is exactly
    // the crossing predicate: no-cross keeps new_off >= stride).
    unsigned fv = row + unsigned(threadIdx.x);
    unsigned fe = (fe_col + unsigned(threadIdx.x))*u_m + unsigned(i0);
    unsigned const step_v = blockDim.x;
    unsigned const step_e = blockDim.x*u_m;
    // Initial chunk resolution only for threads that will actually
    // iterate: a thread with tid >= k_size never enters the loop below,
    // and forming its vbp/ebp anyway can index one slot past a table
    // when fv/fe sit exactly on a chunk multiple (piece-B critic's
    // latent OOB ring-slot read; never dereferenced, now unreachable).
    // (piece F, 2026-10-19) BLOCK-UNIFORM VOCAB FAST PATH: row, and the
    // whole vocab walk [row, row+k_size), is uniform per block (one
    // token per occurrence), and k_size*4 bytes is far below the chunk
    // class, so the walk lies inside ONE vocab chunk unless the row
    // straddles a boundary (measured once per ~5461 rows at k=768). In
    // the common case the vocab base resolves once, uniformly, the
    // loop's vocab advance is a plain pointer add and the mask/rebase
    // branches disappear -- the classic loop body verbatim. The embed
    // side keeps its per-iteration rebase (its per-block spread is
    // m*k elements and crosses chunk boundaries routinely).
    bool const voc_fast = [&]() -> bool {
        if constexpr(VOC1)
        {
            (void)row;
            return true;
        }
        else
        {
            return (row >> VOC_SHIFT)
                == ((row + unsigned(t.k_size) - 1u) >> VOC_SHIFT);
        }
    }();
    T *vbp = nullptr;
    T const *ebp = nullptr;
    if(t.k_size > 0 and unsigned(threadIdx.x) < unsigned(t.k_size))
    {
        if constexpr(VOC1)
        {
            vbp = t.vocab[0] + fv;
        }
        else if(voc_fast)
        {
            // Uniform chunk base (one table read, warp-coalesced) + the
            // ordinary chunk-relative offset; the advance below is then
            // a plain pointer add for the whole walk.
            vbp = t.vocab[row >> VOC_SHIFT] + (fv & VOC_MASK);
        }
        else
        {
            vbp = t.vocab[fv >> VOC_SHIFT] + (fv & VOC_MASK);
        }
        ebp = EMB1
            ? t.embed[0] + fe
            : t.embed[fe >> EMB_SHIFT] + (fe & EMB_MASK);
    }
    for(Index i2 = threadIdx.x; i2 < t.k_size; i2 += Index(step_v))
    {
        const Y contrib = static_cast<Y>(alpha_) * static_cast<Y>(ebp[0]);
        atomicAdd(reinterpret_cast<Z *>(vbp), static_cast<Z>(contrib));
        // Advance with rebase, only when another iteration follows.
        if(i2 + Index(step_v) < t.k_size)
        {
            fv += step_v;
            fe += step_e;
            if constexpr(VOC1)
            {
                vbp = t.vocab[0] + fv;
            }
            else if(voc_fast)
            {
                vbp += step_v;
            }
            else if((fv & VOC_MASK) < step_v)
            {
                vbp = t.vocab[fv >> VOC_SHIFT] + (fv & VOC_MASK);
            }
            else
            {
                vbp += step_v;
            }
            if constexpr(EMB1)
            {
                ebp = t.embed[0] + fe;
            }
            else if((fe & EMB_MASK) < step_e)
            {
                ebp = t.embed[fe >> EMB_SHIFT] + (fe & EMB_MASK);
            }
            else
            {
                ebp += step_e;
            }
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, Index vocab_nelems, Scalar alpha,
        Scalar beta,
        const int64_t *const *index_chunks, Index nindex, Index elems_index,
        const T *const *embed_chunks, Index nembed, Index elems_embed,
        T *const *vocab_chunks, Index nvocab, Index elems_vocab)
    noexcept
{
    if(m <= 0 or n <= 0 or index_range <= 0 or nvocab <= 0)
    {
        return;
    }
    EmbBwdChunkTables<T> t;
    t.index = index_chunks;
    t.embed = embed_chunks;
    t.vocab = vocab_chunks;
    t.nindex = nindex;
    t.nembed = nembed;
    t.nvocab = nvocab;
    t.elems_index = elems_index;
    t.elems_embed = elems_embed;
    t.elems_vocab = elems_vocab;
    t.shift_index = chunk_shift_of<::int64_t>(t.nindex, elems_index, "nntile_embedding_backward");
    t.shift_embed = chunk_shift_of<T>(t.nembed, elems_embed, "nntile_embedding_backward");
    t.shift_vocab = chunk_shift_of<T>(t.nvocab, elems_vocab, "nntile_embedding_backward");
    t.m = m;
    t.n = n;
    t.k = k;
    t.k_start = k_start;
    t.k_size = k_size;
    t.index_range = index_range;
    if(beta == Scalar{0.0})
    {
        // One chunk-aware zero pass (single launch, like the whole-buffer
        // entry's own memset of vocab_nelems). A moderate grid (each thread
        // grid-strides over several 16-byte stores) measures ~565 GB/s vs
        // ~553 for a one-store-per-thread grid and ~577 for a contiguous
        // cudaMemsetAsync; per-chunk memsets are far worse (~440 GB/s from
        // the launch overhead).
        long const total = static_cast<long>(vocab_nelems);
        long const block = 256;
        long const work = (total + 3) / 4; // vector-path work units
        long const grid = (work + block - 1) / block;
        long const grid_cap = 8192;
        long const g = grid < grid_cap ? (grid < 1 ? 1 : grid) : grid_cap;
        (chunked_zero_cuda_kernel<T>)<<<dim3(unsigned(g)), dim3(256), 0,
            stream>>>(t, total);
    }
    dim3 threads(256, 1, 1);
    // U32 fast form: every flat offset the walk forms must fit unsigned
    // (index flat < m*n, embed flat < (k*n + k_start + k_size)*m + m,
    // vocab flat < k_size*index_range + k_size), and the embed stride
    // step (blockDim.x*m) must not overflow the 32-bit fe walk -- the
    // +257*m term covers both emb_span and one step beyond it.
    unsigned long long const idx_span =
        static_cast<unsigned long long>(m)*static_cast<unsigned long long>(n);
    unsigned long long const voc_span =
        static_cast<unsigned long long>(k_size)
        *static_cast<unsigned long long>(index_range)
        + static_cast<unsigned long long>(k_size);
    unsigned long long const emb_span =
        (static_cast<unsigned long long>(k)*static_cast<unsigned long long>(n)
        + static_cast<unsigned long long>(k_start)
        + static_cast<unsigned long long>(k_size)
        + 257ull)*static_cast<unsigned long long>(m)
        + static_cast<unsigned long long>(m);
    bool const u32 = idx_span < (1ull << 32) and voc_span < (1ull << 32)
        and emb_span < (1ull << 32);
    if(u32)
    {
        bool const idx1 = t.nindex <= 1;
        bool const emb1 = t.nembed <= 1;
        bool const voc1 = t.nvocab <= 1;
        // One 256-thread block per (i0, i1); lanes walk the k axis in
        // blockDim.x strides (the classic per-instruction patterns).
        dim3 const sblocks(1, unsigned(m), unsigned(n));
#define NNHAUL_EMBBWD_LAUNCH(I1, E1, V1) \
    (chunked_strided_cuda_kernel<T, I1, E1, V1>)<<<sblocks, threads, 0, \
        stream>>>(t, alpha)
        if(idx1)
        {
            if(emb1)
            {
                if(voc1) { NNHAUL_EMBBWD_LAUNCH(true, true, true); }
                else { NNHAUL_EMBBWD_LAUNCH(true, true, false); }
            }
            else
            {
                if(voc1) { NNHAUL_EMBBWD_LAUNCH(true, false, true); }
                else { NNHAUL_EMBBWD_LAUNCH(true, false, false); }
            }
        }
        else
        {
            if(emb1)
            {
                if(voc1) { NNHAUL_EMBBWD_LAUNCH(false, true, true); }
                else { NNHAUL_EMBBWD_LAUNCH(false, true, false); }
            }
            else
            {
                if(voc1) { NNHAUL_EMBBWD_LAUNCH(false, false, true); }
                else { NNHAUL_EMBBWD_LAUNCH(false, false, false); }
            }
        }
#undef NNHAUL_EMBBWD_LAUNCH
        return;
    }
    dim3 blocks((k_size+255)/256, m, n);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(t, alpha);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range, Index vocab_nelems,
        Scalar alpha,
        Scalar beta, const int64_t *index, const fp32_t *embed, fp32_t *vocab)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range, Index vocab_nelems,
        Scalar alpha,
        Scalar beta, const int64_t *index, const fp64_t *embed, fp64_t *vocab)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range, Index vocab_nelems,
        Scalar alpha,
        Scalar beta, const int64_t *index, const bf16_t *embed, bf16_t *vocab)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range, Index vocab_nelems,
        Scalar alpha,
        Scalar beta, const int64_t *index, const fp16_t *embed, fp16_t *vocab)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range, Index vocab_nelems,
        Scalar alpha, Scalar beta,
        const int64_t *const *index_chunks, Index nindex, Index elems_index,
        const fp32_t *const *embed_chunks, Index nembed, Index elems_embed,
        fp32_t *const *vocab_chunks, Index nvocab, Index elems_vocab)
    noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range, Index vocab_nelems,
        Scalar alpha, Scalar beta,
        const int64_t *const *index_chunks, Index nindex, Index elems_index,
        const fp64_t *const *embed_chunks, Index nembed, Index elems_embed,
        fp64_t *const *vocab_chunks, Index nvocab, Index elems_vocab)
    noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range, Index vocab_nelems,
        Scalar alpha, Scalar beta,
        const int64_t *const *index_chunks, Index nindex, Index elems_index,
        const bf16_t *const *embed_chunks, Index nembed, Index elems_embed,
        bf16_t *const *vocab_chunks, Index nvocab, Index elems_vocab)
    noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range, Index vocab_nelems,
        Scalar alpha, Scalar beta,
        const int64_t *const *index_chunks, Index nindex, Index elems_index,
        const fp16_t *const *embed_chunks, Index nembed, Index elems_embed,
        fp16_t *const *vocab_chunks, Index nvocab, Index elems_vocab)
    noexcept;

} // namespace nntile::kernel::embedding_backward
