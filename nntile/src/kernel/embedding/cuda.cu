/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/embedding/cuda.cu
 * Embeddings from vocabulary within buffers
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/embedding/cuda.hh"
#include <algorithm>
#include <cstring>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::embedding
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index k_start, Index k_size,
        Index index_range, const Index *index, const T *vocab, T *embed)
//! Fill embedding from vocabulary
/*! Fill provided m-by-k-by-n output tensor embed:
 *      embed[i, k_start:k_start+k_size, j] = vocab[:, index[i, j]]
 *
 * @param[in] m: Size of the first mode of index and embed tensors
 * @param[in] n: Size of the last mode of index and embed tensors
 * @param[in] k: Size of the middle mode of embed tensor
 * @param[in] k_start: Offset of the middle mode of embed tensor
 * @param[in] k_size: Size of the first mode of vocab tensor
 * @param[in] index: Tokens (indices of embeddings)
 * @param[in] vocab: Vocabulary of embeddings. It is a contiguous matrix of shape
 *      (k_size, vocab_size) but vocab_size is not passed as a parameter.
 * @param[inout] embed: Output tensor to be filled with embeddings
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    if(i2 < k_size and i1 < n and i0 < m)
    {
            // The token becomes a raw pointer offset below: validate it
            // first and trap loudly (like a PyTorch device assert)
            // instead of corrupting memory.
            const long long token = static_cast<long long>(index[i1*m+i0]);
            if(token < 0 or token >= static_cast<long long>(index_range))
            {
                printf("nntile embedding: token id %lld is outside the "
                        "vocabulary range [0, %lld)\n", token,
                        static_cast<long long>(index_range));
                __trap();
            }
            // Input slice of vocabulary
            const T *vocab_slice = vocab + k_size*token;
            // Output value to be updated
            embed[(i1*k+k_start+i2)*m + i0] = vocab_slice[i2];
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, const int64_t *index_,
        const T *vocab, T *embed)
    noexcept
//! Fill embedding from vocabulary
/*! Fill provided m-by-k-by-n output tensor embed:
 *      embed[i, k_start:k_start+k_size, j] = vocab[:, index[i, j]]
 *
 * @param[in] m: Size of the first mode of index and embed tensors
 * @param[in] n: Size of the last mode of index and embed tensors
 * @param[in] k: Size of the middle mode of embed tensor
 * @param[in] k_start: Offset of the middle mode of embed tensor
 * @param[in] k_size: Size of the first mode of vocab tensor
 * @param[in] index_: Tokens (indices of embeddings)
 * @param[in] vocab: Vocabulary of embeddings. It is a contiguous matrix of shape
 *      (k_size, vocab_size) but vocab_size is not passed as a parameter.
 * @param[inout] embed: Output tensor to be filled with embeddings
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
            std::min(int(k_size), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
            (k_size+threads.z-1)/threads.z);
    using I = typename CUDAComputeType<int64_t>::value;
    auto index = reinterpret_cast<const I *>(index_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, k_start,
            k_size, index_range, index, vocab, embed);
}

//! Chunk descriptors for the one-launch chunked embedding gather (cuda.hh).
template<typename T>
struct EmbChunkTables
{
    const int64_t *const *index; //!< device-ring view of the index table
    const T *const *vocab; //!< device-ring view of the vocab table
    T *const *embed; //!< device-ring view of the embed table
    Index nindex;
    Index nvocab;
    Index nembed;
    Index elems_index; //!< elements in a full index chunk
    Index elems_vocab; //!< elements in a full vocab chunk
    Index elems_embed; //!< elements in a full embed chunk
    int shift_index; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_vocab; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_embed; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    Index m;
    Index n;
    Index k;
    Index k_start;
    Index k_size;
    Index index_range;
};

//! Chunk index of a flat element offset (power-of-two shift when
//! possible; the single-chunk sentinel 62 maps everything to chunk 0).
static __device__ __forceinline__ Index chunk_of(long f, Index elems,
    int shift)
{
    (void)elems;
    return Index(f >> shift);
}

//! Chunk-aware embedding gather: the same (i0, i1, i2) thread geometry as
//! cuda_kernel above, but each of the three operands is addressed through
//! its chunk table. Piece 9 (rule 10): NO shared-memory staging -- every
//! thread reads its chunk pointer straight from the device ring (the
//! tables are a handful of L1-resident entries); the piece-7 staging
//! predates the rule-7 compile-time shifts and was a workaround for
//! runtime shifts. The per-operand chunk shifts stay COMPILE-TIME
//! constants (the IDX1/EMB1/VOC1 template flags pick the sentinel 62 for
//! a 1-chunk operand -- no chunk arithmetic at all for it -- or the
//! rule-7 class shift). U32=true (host-guaranteed when every operand's
//! flat span fits 32 bits) runs the flat arithmetic in 32 bits; U32=false
//! keeps piece-4's compare-first 64-bit global-table body for huge tiles
//! (the ring's no-cap contract). The dispatch is template parameters,
//! never per-element selects. A pure copy with the staged form's own
//! pointer VALUES, so bitwise the whole-buffer call.
template<typename T, bool IDX1, bool EMB1, bool VOC1, bool U32>
static __global__
void chunked_cuda_kernel(EmbChunkTables<T> t)
{
    using I = typename CUDAComputeType<int64_t>::value;
    constexpr std::size_t class_elems = kChunkClassBytes / sizeof(T);
    constexpr int IDX_SHIFT = IDX1 ? kChunkSingleShift
        : chunk_log2(kChunkClassBytes / sizeof(I));
    constexpr int VOC_SHIFT = VOC1 ? kChunkSingleShift
        : chunk_log2(class_elems);
    constexpr int EMB_SHIFT = EMB1 ? kChunkSingleShift
        : chunk_log2(class_elems);
    Index const i0 = threadIdx.x + blockIdx.x*blockDim.x;
    Index const i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index const i2 = threadIdx.z + blockIdx.z*blockDim.z;
    if(i2 < t.k_size and i1 < t.n and i0 < t.m)
    {
        if(U32)
        {
            // 32-bit flat offsets with compile-time shifts; 1-chunk
            // operands resolve to their table slot-0 pointer with no
            // chunk arithmetic, read directly from the ring (rule 10).
            unsigned const u_i0 = unsigned(i0), u_i1 = unsigned(i1),
                u_i2 = unsigned(i2);
            unsigned const u_m = unsigned(t.m),
                u_k_size = unsigned(t.k_size);
            unsigned const u_fi = u_i1*u_m + u_i0;
            I const *ip = IDX1
                ? reinterpret_cast<I const *>(t.index[0]) + u_fi
                : reinterpret_cast<I const *>(t.index[u_fi >> IDX_SHIFT])
                    + (u_fi & ((1u << IDX_SHIFT) - 1u));
            const long long token = static_cast<long long>(ip[0]);
            if(token < 0 or token >= static_cast<long long>(t.index_range))
            {
                printf("nntile embedding: token id %lld is outside the "
                        "vocabulary range [0, %lld)\n", token,
                        static_cast<long long>(t.index_range));
                __trap();
            }
            unsigned const fv = u_k_size*unsigned(token) + u_i2;
            T const *vp = VOC1
                ? t.vocab[0] + fv
                : t.vocab[fv >> VOC_SHIFT]
                    + (fv & ((1u << VOC_SHIFT) - 1u));
            unsigned const fe = unsigned(i1)*unsigned(t.k)
                + unsigned(t.k_start) + u_i2;
            T *ep = EMB1
                ? t.embed[0] + (fe*u_m + u_i0)
                : t.embed[(fe*u_m + u_i0) >> EMB_SHIFT]
                    + ((fe*u_m + u_i0) & ((1u << EMB_SHIFT) - 1u));
            ep[0] = vp[0];
        }
        else
        {
            // Piece 4's compare-first resolution against the global ring
            // tables (single-chunk operands resolve with one compare --
            // table[0] + f -- so the chunk arithmetic runs only for
            // genuinely multi-chunk operands). Same addresses either way.
            long const fi = long(i1) * long(t.m) + long(i0);
            I const *ip;
            if(fi < t.elems_index)
            {
                ip = reinterpret_cast<I const *>(t.index[0]) + fi;
            }
            else
            {
                Index const ci = chunk_of(fi, t.elems_index, t.shift_index);
                ip = reinterpret_cast<I const *>(t.index[ci])
                    + (fi - long(ci) * long(t.elems_index));
            }
            const long long token = static_cast<long long>(ip[0]);
            if(token < 0 or token >= static_cast<long long>(t.index_range))
            {
                printf("nntile embedding: token id %lld is outside the "
                        "vocabulary range [0, %lld)\n", token,
                        static_cast<long long>(t.index_range));
                __trap();
            }
            long const fv = long(t.k_size) * long(token) + long(i2);
            T const *vp;
            if(fv < t.elems_vocab)
            {
                vp = t.vocab[0] + fv;
            }
            else
            {
                Index const cv = chunk_of(fv, t.elems_vocab, t.shift_vocab);
                vp = t.vocab[cv]
                    + (fv - long(cv) * long(t.elems_vocab));
            }
            long const fe = (long(i1) * long(t.k) + long(t.k_start)
                + long(i2)) * long(t.m) + long(i0);
            T *ep;
            if(fe < t.elems_embed)
            {
                ep = t.embed[0] + fe;
            }
            else
            {
                Index const ce = chunk_of(fe, t.elems_embed, t.shift_embed);
                ep = t.embed[ce]
                    + (fe - long(ce) * long(t.elems_embed));
            }
            ep[0] = vp[0];
        }
    }
}

//! Run-form chunk-aware embedding gather (piece B, 2026-10-18c). Each
//! thread owns ONE token and NVEC consecutive 16-byte vocab vectors
//! along the k axis (a vocab row is contiguous along k, so the random
//! row read becomes one wide load per 16/sizeof(T) elements instead of
//! one 4-byte load per element), and the token read, its validation and
//! the ring chunk reads amortize over the whole run -- 1/16th the
//! dependent-load chains of the elementwise form on the graph shape.
//! The embed store addresses (column-major: stride m between consecutive
//! k elements; lanes along i0 so every store instruction is one
//! coalesced run) are computed between the vector-load issue and the
//! stores: pointer work interlaced with the loads, never a serialized
//! pointer pass. A 16-byte vector that would straddle a full-chunk
//! boundary takes the per-element fallback -- UNREACHABLE as dispatched
//! (VW always divides the class element count 2^22/2^21/2^23 for
//! 4/8/2-byte types, and k_size % VW == 0 keeps run starts vector
//! aligned, so a vector's in-chunk offset + VW can never exceed the
//! chunk); it stays as the safety net for a future class change.
//! Dispatched only when every operand's flat span fits 32 bits AND
//! k_size is a multiple of the vector width (rows and runs stay
//! 16-byte aligned; chunk bases are >=256-byte aligned), so no vocab
//! row vector straddles allocations; other shapes keep the elementwise
//! kernel above. A pure byte copy: bitwise the classic kernel.
template<typename T, bool IDX1, bool EMB1, bool VOC1>
static __global__
void chunked_run_cuda_kernel(EmbChunkTables<T> t)
{
    using I = typename CUDAComputeType<int64_t>::value;
    constexpr std::size_t class_elems = kChunkClassBytes / sizeof(T);
    constexpr int VW = int(16 / sizeof(T)); //!< elements per 16-byte vector
    constexpr int NVEC = 4; //!< vectors per thread (the piece-B lab's V4)
    constexpr int IDX_SHIFT = IDX1 ? kChunkSingleShift
        : chunk_log2(kChunkClassBytes / sizeof(I));
    constexpr int VOC_SHIFT = VOC1 ? kChunkSingleShift
        : chunk_log2(class_elems);
    constexpr int EMB_SHIFT = EMB1 ? kChunkSingleShift
        : chunk_log2(class_elems);
    constexpr unsigned VOC_MASK = VOC1 ? 0xffffffffu
        : (1u << VOC_SHIFT) - 1u;
    constexpr unsigned VOC_END = VOC1 ? 0xffffffffu : 1u << VOC_SHIFT;
    Index const i0 = threadIdx.x + blockIdx.x*blockDim.x;
    Index const i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index const i2 = (threadIdx.z + blockIdx.z*blockDim.z)
        * Index(VW*NVEC);
    if(i0 >= t.m or i1 >= t.n or i2 >= t.k_size)
    {
        return;
    }
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
        printf("nntile embedding: token id %lld is outside the "
                "vocabulary range [0, %lld)\n", token,
                static_cast<long long>(t.index_range));
        __trap();
    }
    unsigned const fv = unsigned(t.k_size)*unsigned(token) + unsigned(i2);
    unsigned const fe0 = unsigned(i1)*unsigned(t.k) + unsigned(t.k_start)
        + unsigned(i2);
    // Vector loads first (independent -> memory-level parallelism), the
    // per-vector chunk rebase interlaced between the load issues; the
    // rare boundary-straddling vector falls back to element loads.
    unsigned fvv[NVEC];
    uint4 raw[NVEC];
#pragma unroll
    for(int v = 0; v < NVEC; ++v)
    {
        if(i2 + v*VW >= t.k_size)
        {
            break;
        }
        fvv[v] = fv + unsigned(v*VW);
        if constexpr(VOC1)
        {
            raw[v] = *reinterpret_cast<uint4 const *>(t.vocab[0] + fvv[v]);
        }
        else if((fvv[v] & VOC_MASK) + unsigned(VW) <= VOC_END)
        {
            raw[v] = *reinterpret_cast<uint4 const *>(
                t.vocab[fvv[v] >> VOC_SHIFT] + (fvv[v] & VOC_MASK));
        }
        else
        {
            T tmp[VW];
            for(int e = 0; e < VW; ++e)
            {
                unsigned const f = fvv[v] + unsigned(e);
                tmp[e] = t.vocab[f >> VOC_SHIFT][f & VOC_MASK];
            }
            std::memcpy(&raw[v], tmp, sizeof(raw[v]));
        }
    }
    // Stores: the embed address math (per element when multi-chunk)
    // sits between the load issue and the stores.
#pragma unroll
    for(int v = 0; v < NVEC; ++v)
    {
#pragma unroll
        for(int e = 0; e < VW; ++e)
        {
            Index const ie = i2 + v*VW + e;
            if(ie >= t.k_size)
            {
                return;
            }
            unsigned const fflat = (fe0 + unsigned(v*VW + e))*u_m
                + unsigned(i0);
            T *ep = EMB1
                ? t.embed[0] + fflat
                : t.embed[fflat >> EMB_SHIFT]
                    + (fflat & ((1u << EMB_SHIFT) - 1u));
            T val;
            std::memcpy(&val, reinterpret_cast<unsigned char const *>(
                &raw[v]) + e*sizeof(T), sizeof(T));
            ep[0] = val;
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range,
        const int64_t *const *index_chunks, Index nindex, Index elems_index,
        const T *const *vocab_chunks, Index nvocab, Index elems_vocab,
        T *const *embed_chunks, Index nembed, Index elems_embed)
    noexcept
{
    if(m <= 0 or n <= 0 or index_range <= 0)
    {
        return;
    }
    EmbChunkTables<T> t;
    t.index = index_chunks;
    t.vocab = vocab_chunks;
    t.embed = embed_chunks;
    t.nindex = nindex;
    t.nvocab = nvocab;
    t.nembed = nembed;
    t.elems_index = elems_index;
    t.elems_vocab = elems_vocab;
    t.elems_embed = elems_embed;
    t.shift_index = chunk_shift_of<::int64_t>(t.nindex, elems_index, "nntile_embedding");
    t.shift_vocab = chunk_shift_of<T>(t.nvocab, elems_vocab, "nntile_embedding");
    t.shift_embed = chunk_shift_of<T>(t.nembed, elems_embed, "nntile_embedding");
    t.m = m;
    t.n = n;
    t.k = k;
    t.k_start = k_start;
    t.k_size = k_size;
    t.index_range = index_range;
    // Same thread geometry as the whole-buffer entry.
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
            std::min(int(k_size), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
            (k_size+threads.z-1)/threads.z);
    bool const idx1 = t.nindex <= 1;
    bool const emb1 = t.nembed <= 1;
    bool const voc1 = t.nvocab <= 1;
    // U32 fast instantiation: every flat offset the kernel forms must fit
    // unsigned (index flat < m*n, vocab flat < k_size*index_range + k_size,
    // embed flat < (k*n + k_start + k_size)*m + m).
    unsigned long long const idx_span =
        static_cast<unsigned long long>(m)*static_cast<unsigned long long>(n);
    unsigned long long const voc_span =
        static_cast<unsigned long long>(k_size)
        *static_cast<unsigned long long>(index_range)
        + static_cast<unsigned long long>(k_size);
    unsigned long long const emb_span =
        (static_cast<unsigned long long>(k)*static_cast<unsigned long long>(n)
        + static_cast<unsigned long long>(k_start)
        + static_cast<unsigned long long>(k_size))
        *static_cast<unsigned long long>(m)
        + static_cast<unsigned long long>(m);
    bool const u32 = idx_span < (1ull << 32) and voc_span < (1ull << 32)
        and emb_span < (1ull << 32);
    // Run-form fast path: k_size a multiple of the 16-byte vector width
    // keeps every vocab-row vector (and its 16-byte alignment) inside
    // one chunk/row; other shapes keep the elementwise body.
    int const vw = int(16 / sizeof(T));
    // (review-540 v3 #1) The run-form launch also needs n <= 65535
    // (gridDim.y) and blocks.z <= 64; shapes beyond fall through to the
    // elementwise body, which serves them exactly (the classic path's
    // coverage is preserved -- no abort).
    unsigned const vec_n = unsigned(t.n);
    unsigned const vec_z =
        (unsigned(t.k_size) / unsigned(vw) + 8u * 4u - 1u) / (8u * 4u);
    bool const vec = u32 and t.k_size % Index(vw) == 0
        and vec_n <= 65535u and vec_z <= 64u;
    if(vec)
    {
        // Same operand-chunk flag cross, run-form bodies: threads.x
        // along m (lanes at consecutive i0 -> coalesced embed stores),
        // threads.z over runs of NVEC vectors along k_size.
        unsigned const tx = t.m < 32 ? unsigned(t.m) : 32u;
        dim3 const threads(tx, 1, 8);
        unsigned const vectors = unsigned(t.k_size) / unsigned(vw);
        unsigned const vpb = 8u * 4u; // threads.z * NVEC vectors per block
        dim3 const blocks((unsigned(t.m) + tx - 1u)/tx, unsigned(t.n),
            (vectors + vpb - 1u)/vpb);
#define NNHAUL_EMB_RUN_LAUNCH(I1, E1, V1) \
    (chunked_run_cuda_kernel<T, I1, E1, V1>)<<<blocks, threads, 0, \
        stream>>>(t)
        if(idx1)
        {
            if(emb1)
            {
                if(voc1) { NNHAUL_EMB_RUN_LAUNCH(true, true, true); }
                else { NNHAUL_EMB_RUN_LAUNCH(true, true, false); }
            }
            else
            {
                if(voc1) { NNHAUL_EMB_RUN_LAUNCH(true, false, true); }
                else { NNHAUL_EMB_RUN_LAUNCH(true, false, false); }
            }
        }
        else
        {
            if(emb1)
            {
                if(voc1) { NNHAUL_EMB_RUN_LAUNCH(false, true, true); }
                else { NNHAUL_EMB_RUN_LAUNCH(false, true, false); }
            }
            else
            {
                if(voc1) { NNHAUL_EMB_RUN_LAUNCH(false, false, true); }
                else { NNHAUL_EMB_RUN_LAUNCH(false, false, false); }
            }
        }
#undef NNHAUL_EMB_RUN_LAUNCH
        return;
    }
    if(u32)
    {
        // Direct-ring instantiation cross (8 kernels): the flags pick the
        // compile-time shifts; U32=true is the fast path.
#define NNHAUL_EMB_LAUNCH(I1, E1, V1) \
    (chunked_cuda_kernel<T, I1, E1, V1, true>)<<<blocks, threads, 0, \
        stream>>>(t)
        if(idx1)
        {
            if(emb1)
            {
                if(voc1) { NNHAUL_EMB_LAUNCH(true, true, true); }
                else { NNHAUL_EMB_LAUNCH(true, true, false); }
            }
            else
            {
                if(voc1) { NNHAUL_EMB_LAUNCH(true, false, true); }
                else { NNHAUL_EMB_LAUNCH(true, false, false); }
            }
        }
        else
        {
            if(emb1)
            {
                if(voc1) { NNHAUL_EMB_LAUNCH(false, true, true); }
                else { NNHAUL_EMB_LAUNCH(false, true, false); }
            }
            else
            {
                if(voc1) { NNHAUL_EMB_LAUNCH(false, false, true); }
                else { NNHAUL_EMB_LAUNCH(false, false, false); }
            }
        }
#undef NNHAUL_EMB_LAUNCH
    }
    else
    {
        (chunked_cuda_kernel<T, false, false, false, false>)<<<blocks,
            threads, 0, stream>>>(t);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range,
        const int64_t *index, const fp32_t *vocab, fp32_t *embed)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range,
        const int64_t *index, const bf16_t *vocab, bf16_t *embed)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range,
        const int64_t *index, const fp16_t *vocab, fp16_t *embed)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range,
        const int64_t *index, const fp64_t *vocab, fp64_t *embed)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range,
        const int64_t *const *index_chunks, Index nindex, Index elems_index,
        const fp32_t *const *vocab_chunks, Index nvocab, Index elems_vocab,
        fp32_t *const *embed_chunks, Index nembed, Index elems_embed)
    noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range,
        const int64_t *const *index_chunks, Index nindex, Index elems_index,
        const fp64_t *const *vocab_chunks, Index nvocab, Index elems_vocab,
        fp64_t *const *embed_chunks, Index nembed, Index elems_embed)
    noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range,
        const int64_t *const *index_chunks, Index nindex, Index elems_index,
        const bf16_t *const *vocab_chunks, Index nvocab, Index elems_vocab,
        bf16_t *const *embed_chunks, Index nembed, Index elems_embed)
    noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range,
        const int64_t *const *index_chunks, Index nindex, Index elems_index,
        const fp16_t *const *vocab_chunks, Index nvocab, Index elems_vocab,
        fp16_t *const *embed_chunks, Index nembed, Index elems_embed)
    noexcept;

} // namespace nntile::kernel::embedding
