/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/randn/cpu.cc
 * Randn operation on a buffer on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/randn/cpu.hh"
#include <cmath>
#include <iostream>
#include "../external/random.h" // from external
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::randn
{

static inline float chameleon_randn(unsigned long long &seed, float mean,
        float stddev)
{
    constexpr float two=2.0, twopi=6.2831853071795864769252867663;
    float t1 = CORE_slaran(&seed);
    float t2 = CORE_slaran(&seed) * twopi;
    float t3 = std::sqrt(-two*std::log(t1)) * std::cos(t2);
    return stddev*t3 + mean;
}

static inline double chameleon_randn(unsigned long long &seed, double mean,
        double stddev)
{
    constexpr double two=2.0, twopi=6.2831853071795864769252867663;
    double t1 = CORE_dlaran(&seed);
    double t2 = CORE_dlaran(&seed) * twopi;
    double t3 = std::sqrt(-two*std::log(t1)) * std::cos(t2);
    return stddev*t3 + mean;
}

template<typename T>
void cpu(Index ndim, Index nelems, unsigned long long seed,
        Scalar mean_, Scalar stddev_, const Index *start, const Index *shape,
        const Index *underlying_shape, T *data, const Index *stride,
        int64_t *tmp_index_)
    noexcept
//! Fill manydimensional array with random normally distributed numbers
/*! The output is generated as if it is a part of another many-dimensional
 * array, also called the underlying array. The main reason to treat the
 * output and the underlying arrays separately is to make it possible to
 * fill randomly any contiguous subarray on demand. Contiguous subarray is such
 * a subarray that is based on contiguous range of indexes in each dimension.
 * In the end, this routine makes it possible to randomly generate any array in
 * parallel by parts. These parts are not obliged to belong to a single memory
 * buffer and can be stored separately.
 *
 * Mnemonically, this routine randomly fills output buffer as if entire
 * underlying array was at first generated with a provided seed and then copied
 * output=underlying[start:start+shape].
 *
 * @param[in] ndim: Number of dimensions of the output array
 * @param[in] nelems: Number of elements of the output array
 * @param[in] seed: Random seed for the entire underlying array
 * @param[in] mean_: Average value of the normal distribution
 * @param[in] stddev_: Standard deviation of the normal distribution
 * @param[in] start: Starting index of a subarray to generate. Contains ndim
 *      values.
 * @param[in] shape: Shape of the output array. Contains ndim values.
 * @param[in] underlying_shape: Shape of the underlying array. Contains ndim
 *      values.
 * @param[out] data: The output array memory buffer
 * @param[in] stride: Strides of the output array. Contains ndim values.
 * @param[scratch] tmp_index_: Temporary buffer for indexing purposes. Contains
 *      ndim values.
 * */
{
    using Y = typename T::repr_t;
    // auto data = reinterpret_cast<Y *>(data_);
    Y mean{mean_}, stddev{stddev_};
    using I = typename CPUComputeType<int64_t>::value;
    auto tmp_index = reinterpret_cast<I *>(tmp_index_);
    // Jump to the first element to generate
    Index shift = start[ndim-1];
    for(Index i = ndim-2; i >= 0; --i)
    {
        shift = start[i] + shift*underlying_shape[i];
    }
    seed = CORE_rnd64_jump(shift, seed);
    // View tile as a matrix of shape (shape[0], prod(shape[1:ndim]))
    Index nrows = shape[0], ncols = nelems / nrows;
    // Generate the first column
    for(Index i = 0; i < nrows; ++i)
    {
        *data = static_cast<T>(chameleon_randn(seed, mean, stddev));
        data += stride[0];
    }
    // Init temporary index
    for(Index i = 0; i < ndim; ++i)
    {
        tmp_index[i] = 0;
    }
    // Generate all other columns
    for(Index j = 1; j < ncols; ++j)
    {
        // Get index of the first element of the current column as well as a
        // shift to it from the last generated element. Init the index by
        // incrementing it properly (ignore 0 dimension, as it is a row index).
        ++tmp_index[1];
        Index k = 1;
        // Init shift
        Index shift = underlying_shape[0] - nrows;
        // Init stride for the current dimension
        Index underlying_stride = underlying_shape[0];
        // Update pointer to the current buffer element
        data += stride[1] - stride[0]*nrows;
        // Check if currently stored index is out-of-bounds
        while(tmp_index[k] == shape[k])
        {
            // Reset out-of-bound index
            tmp_index[k] = 0;
            // Increment next index
            ++k;
            ++tmp_index[k];
            // Update shift
            shift += underlying_stride * (underlying_shape[k-1]-shape[k-1]);
            // Update stride for the current dimension
            underlying_stride *= underlying_shape[k-1];
            // Update pointer to the current buffer element
            data += stride[k] - stride[k-1]*shape[k-1];
        }
        // Now both the index of the first element of the current column and
        // the shift are ready. At first jump random generator shift times.
        seed = CORE_rnd64_jump(shift, seed);
        // Generate the current column
        for(Index i = 0; i < nrows; ++i)
        {
            *data = static_cast<T>(chameleon_randn(seed, mean, stddev));
            data += stride[0];
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index ndim, Index nelems, unsigned long long seed,
        Scalar mean, Scalar stddev, const Index *start, const Index *shape,
        const Index *underlying_shape, fp32_t *data, const Index *stride,
        int64_t *tmp_index)
    noexcept;

template
void cpu<fp64_t>(Index ndim, Index nelems, unsigned long long seed,
        Scalar mean, Scalar stddev, const Index *start, const Index *shape,
        const Index *underlying_shape, fp64_t *data, const Index *stride,
        int64_t *tmp_index)
    noexcept;

template
void cpu<bf16_t>(Index ndim, Index nelems, unsigned long long seed,
        Scalar mean, Scalar stddev, const Index *start, const Index *shape,
        const Index *underlying_shape, bf16_t *data, const Index *stride,
        int64_t *tmp_index)
    noexcept;

template<typename T>
void cpu_ndim0(unsigned long long seed, Scalar mean_, Scalar stddev_, T *data)
    noexcept
{
    // 0-dimensional tensor is just a scalar
    using Y = typename T::repr_t;
    // auto data = reinterpret_cast<Y *>(data_);
    Y mean{mean_}, stddev{stddev_};
    *data = static_cast<T>(chameleon_randn(seed, mean, stddev));
}

// Explicit instantiation
template
void cpu_ndim0<fp32_t>(unsigned long long seed, Scalar mean, Scalar stddev,
        fp32_t *data);

template
void cpu_ndim0<bf16_t>(unsigned long long seed, Scalar mean, Scalar stddev,
        bf16_t *data);

template
void cpu_ndim0<fp64_t>(unsigned long long seed, Scalar mean, Scalar stddev,
        fp64_t *data);

} // namespace nntile::kernel::randn
