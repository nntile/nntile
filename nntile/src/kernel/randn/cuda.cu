/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/randn/cuda.cu
 * Randn operation on CUDA (FP32 product path; same chameleon PRNG as CPU)
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/randn/cuda.hh"
#include "nntile/kernel/cuda.hh"
#include <cmath>

namespace nntile::kernel::randn
{

// Chameleon LCG from external/random.h, inlined for device.
#define Rnd64_A 6364136223846793005ULL
#define Rnd64_C 1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20

static constexpr int MAX_NDIM = 8;

struct RandnGeom
{
    Index ndim;
    Index start[MAX_NDIM];
    Index shape[MAX_NDIM];
    Index stride[MAX_NDIM];
    Index underlying_shape[MAX_NDIM];
};

__device__ inline unsigned long long
device_rnd64_jump(unsigned long long n, unsigned long long seed)
{
    unsigned long long a_k = Rnd64_A;
    unsigned long long c_k = Rnd64_C;
    unsigned long long ran = seed;
    n <<= 1;
    for(int i = 0; n; n >>= 1, ++i)
    {
        if(n & 1)
        {
            ran = a_k * ran + c_k;
        }
        c_k *= (a_k + 1);
        a_k *= a_k;
    }
    return ran;
}

__device__ inline float device_slaran(unsigned long long &ran)
{
    float value = ran * RndF_Mul;
    ran = Rnd64_A * ran + Rnd64_C;
    return value;
}

__device__ inline double device_dlaran(unsigned long long &ran)
{
    double value = ran * RndD_Mul;
    ran = Rnd64_A * ran + Rnd64_C;
    return value;
}

__device__ inline float
device_chameleon_randn(unsigned long long &seed, float mean, float stddev)
{
    constexpr float two = 2.0f, twopi = 6.2831853071795864769252867663f;
    float t1 = device_slaran(seed);
    float t2 = device_slaran(seed) * twopi;
    float t3 = sqrtf(-two * logf(t1)) * cosf(t2);
    return stddev * t3 + mean;
}

__device__ inline double
device_chameleon_randn(unsigned long long &seed, double mean, double stddev)
{
    constexpr double two = 2.0, twopi = 6.2831853071795864769252867663;
    double t1 = device_dlaran(seed);
    double t2 = device_dlaran(seed) * twopi;
    double t3 = sqrt(-two * log(t1)) * cos(t2);
    return stddev * t3 + mean;
}

template<typename T>
static __global__
void cuda_kernel(Index nelems, unsigned long long seed, Scalar mean_,
        Scalar stddev_, RandnGeom geom, T *data)
{
    Index i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= nelems)
    {
        return;
    }
    using Y = typename T::repr_t;
    Y mean{mean_}, stddev{stddev_};
    Index ndim = geom.ndim;
    Index lin = i;
    Index offset = 0;
    Index under = 0;
    Index under_stride = 1;
    for(Index d = 0; d < ndim; ++d)
    {
        Index coord = lin % geom.shape[d];
        lin /= geom.shape[d];
        offset += coord * geom.stride[d];
        under += (coord + geom.start[d]) * under_stride;
        under_stride *= geom.underlying_shape[d];
    }
    unsigned long long local = device_rnd64_jump(
        static_cast<unsigned long long>(under), seed);
    data[offset] = static_cast<T>(device_chameleon_randn(local, mean, stddev));
}

template<typename T>
void cuda(cudaStream_t stream, Index ndim, Index nelems,
        unsigned long long seed, Scalar mean, Scalar stddev,
        const Index *start, const Index *shape,
        const Index *underlying_shape, T *data, const Index *stride)
    noexcept
{
    if(ndim <= 0 || ndim > MAX_NDIM || nelems <= 0)
    {
        return;
    }
    RandnGeom geom{};
    geom.ndim = ndim;
    for(Index d = 0; d < ndim; ++d)
    {
        geom.start[d] = start[d];
        geom.shape[d] = shape[d];
        geom.stride[d] = stride[d];
        geom.underlying_shape[d] = underlying_shape[d];
    }
    dim3 threads(256);
    dim3 blocks((nelems + 255) / 256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(
        nelems, seed, mean, stddev, geom, data);
}

template
void cuda<fp32_t>(cudaStream_t stream, Index ndim, Index nelems,
        unsigned long long seed, Scalar mean, Scalar stddev,
        const Index *start, const Index *shape,
        const Index *underlying_shape, fp32_t *data, const Index *stride)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index ndim, Index nelems,
        unsigned long long seed, Scalar mean, Scalar stddev,
        const Index *start, const Index *shape,
        const Index *underlying_shape, fp64_t *data, const Index *stride)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index ndim, Index nelems,
        unsigned long long seed, Scalar mean, Scalar stddev,
        const Index *start, const Index *shape,
        const Index *underlying_shape, fp32_fast_tf32_t *data,
        const Index *stride)
    noexcept;

template
void cuda<fp32_fast_fp16_t>(cudaStream_t stream, Index ndim, Index nelems,
        unsigned long long seed, Scalar mean, Scalar stddev,
        const Index *start, const Index *shape,
        const Index *underlying_shape, fp32_fast_fp16_t *data,
        const Index *stride)
    noexcept;

template
void cuda<fp32_fast_bf16_t>(cudaStream_t stream, Index ndim, Index nelems,
        unsigned long long seed, Scalar mean, Scalar stddev,
        const Index *start, const Index *shape,
        const Index *underlying_shape, fp32_fast_bf16_t *data,
        const Index *stride)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index ndim, Index nelems,
        unsigned long long seed, Scalar mean, Scalar stddev,
        const Index *start, const Index *shape,
        const Index *underlying_shape, bf16_t *data, const Index *stride)
    noexcept;

} // namespace nntile::kernel::randn
