/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/base_types.hh
 * Base integer and floating point types.
 *
 * @version 1.0.0
 * */

#pragma once

#include <cstdint>
#include <assert.h>
#include <iostream>

namespace nntile
{

//! Enumeration of base supported data types inside tiles and tensors
enum class DTypeEnum: int
{
    NOT_INITIALIZED=101,
    INT64,
    INT32,
    INT16,
    INT8,
    BOOL,
    FP64,
    FP32,
    FP32_FAST_TF32,
    FP32_FAST_FP16,
    FP32_FAST_BF16,
    TF32,
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2
};
//    constexpr char *name()
//    {
//        switch(value)
//        {
//            case INT64:
//                return "int64";
//                break;
//            case INT32:
//                return "int32";
//                break;
//            case INT16:
//                return "int16";
//                break;
//            case INT8:
//                return "int8";
//                break;
//            case BOOL:
//                return "bool";
//                break;
//            case FP64:
//                return "fp64";
//                break;
//            case FP32:
//                return "fp32";
//                break;
//            case FP32_FAST_TF32:
//                return "fp32_fast_tf32";
//                break;
//            case FP32_FAST_FP16:
//                return "fp32_fast_fp16";
//                break;
//            case FP32_FAST_BF16:
//                return "fp32_fast_bf16";
//                break;
//            case TF32:
//                return "tf32";
//                break;
//            case FP16:
//                return "fp16";
//                break;
//            case BF16:
//                return "bf16";
//                break;
//            case FP8_E4M3:
//                return "fp8_e4m3";
//                break;
//            case FP8_E5M2:
//                return "fp8_e5m2";
//                break;
//            default:
//                return "error";
//        }
//    }
//};

//! This type is meant for scalar values outside StarPU buffers
using scal_t = float;

//! Converter function with static cast
template<typename T, typename Y>
inline constexpr Y T2Y(const T &other)
{
    return static_cast<Y>(other);
}

/*! Type-constructor for custom NNTile types inside StarPU buffers
 *
 * Parameters:
 * @params[in] T: typename of internal value
 * @params[in] nbytes: number of bytes for a singe element of this type
 * @params[in] dtype_enum_: enum description of a type
 * @params[in] T_compat: typename of a CPU-compatible type for IO
 * @params[in] compat2T: conversion function from T_compat type into T type
 * @params[in] T2compat: conversion function from T type into T_compat type
 * */
template<typename T, size_t nbytes, enum DTypeEnum dtype_enum_,
    typename T_compat, T compat2T(const T_compat &)=T2Y<T_compat, T>,
    T_compat T2compat(const T &)=T2Y<T, T_compat>>
class DType
{
public:
    T value;
    using internal_t = T;
    using compat_t = T_compat;
    //using enum = enum DTypeEnum::;
    static constexpr enum DTypeEnum dtype_enum = dtype_enum_;
    explicit constexpr DType(const T_compat &other):
        value(compat2T(other))
    {
        static_assert(sizeof(T) == nbytes);
        static_assert(sizeof(*this) == nbytes);
    }
    constexpr DType &operator=(const T_compat &other)
    {
        value = compat2T(other);
        return *this;
    }
    constexpr DType &operator=(const DType &other)
    {
        value = other.value;
        return *this;
    }
    constexpr T_compat get() const
    {
        return T2compat(value);
    }
    constexpr DType &set(const T_compat &other)
    {
        value = compat2T(other);
        return *this;
    }
    explicit constexpr operator T_compat() const
    {
        return get();
    }
};

//! 64-bit integer type to work inside StarPU buffers
using int64_t = DType<::int64_t, 8, DTypeEnum::INT64, ::int64_t>;

//! 32-bit integer type to work inside StarPU buffers
using int32_t = DType<::int32_t, 4, DTypeEnum::INT32, ::int32_t>;

//! 16-bit integer type to work inside StarPU buffers
using int16_t = DType<::int16_t, 2, DTypeEnum::INT16, ::int16_t>;

//! 8-bit integer type to work inside StarPU buffers
using int8_t = DType<::int8_t, 1, DTypeEnum::INT8, ::int8_t>;

//! Boolean type for mask to work inside StarPU buffers
using bool_t = DType<bool, 1, DTypeEnum::BOOL, bool>;

//! Double precision floating point numbers for StarPU buffers
using fp64_t = DType<double, 8, DTypeEnum::FP64, double>;

//! Single precision floating point numbers for StarPU buffers
using fp32_t = DType<float, 4, DTypeEnum::FP32, float>;

//! Single precision, that relies on TensorFloat type for compute-bound ops
using fp32_fast_tf32_t = DType<float, 4, DTypeEnum::FP32_FAST_TF32, float>;

//! Single precision, that relies on Half type for compute-bound ops
using fp32_fast_fp16_t = DType<float, 4, DTypeEnum::FP32_FAST_FP16, float>;

//! Single precision, that relies on BrainFloat16 type for compute-bound ops
using fp32_fast_bf16_t = DType<float, 4, DTypeEnum::FP32_FAST_BF16, float>;

//! Single precision, that relies on TensorFloat type for all ops
using tf32_t = DType<float, 4, DTypeEnum::TF32, float>;

//! Conversion function from float into fp16_t type
inline constexpr ::int16_t float_to_fp16(const float &other)
{
    // Fake FP32 to FP16 convertor
    ::int16_t res = 0;
    return res;
}

//! Conversion function from fp16_t into float type
inline constexpr float fp16_to_float(const ::int16_t &other)
{
    // Fake FP16 to FP32 convertor
    float res = 0;
    return res;
}

//! Half precision for StarPU buffers
using fp16_t = DType<::int16_t, 2, DTypeEnum::FP16, float, float_to_fp16,
      fp16_to_float>;

//! Conversion function from float into bf16_t type
inline constexpr ::int16_t float_to_bf16(const float &other)
{
    // Fake FP32 to BF16 convertor
    ::int16_t res = 0;
    return res;
}

//! Conversion function from bf16_t into float type
inline constexpr float bf16_to_float(const ::int16_t &other)
{
    // Fake BF16 to FP32 convertor
    float res = 0;
    return res;
}

//! BrainFloat16 precision for StarPU buffers
using bf16_t = DType<::int16_t, 2, DTypeEnum::BF16, float, float_to_bf16,
      bf16_to_float>;

//! Conversion function from float into fp8_e4m3_t type
inline constexpr ::int8_t float_to_e4m3(const float &other)
{
    // Fake FP32 to FP8_E4M3 convertor
    ::int8_t res = 0;
    return res;
}

//! Conversion function from fp8_e4m3_t into float type
inline constexpr float e4m3_to_float(const ::int8_t &other)
{
    // Fake FP8_E4M3 to FP32 convertor
    float res = 0;
    return res;
}

//! FP8_E4M3 precision for StarPU buffers
using fp8_e4m3_t = DType<::int8_t, 1, DTypeEnum::FP8_E4M3, float,
      float_to_e4m3, e4m3_to_float>;

//! Conversion function from float into fp8_e5m2_t type
inline constexpr ::int8_t float_to_e5m2(const float &other)
{
    // Fake FP32 to FP8_E5M2 convertor
    ::int8_t res = 0;
    return res;
}

//! Conversion function from fp8_e5m2_t into float type
inline constexpr float e5m2_to_float(const ::int8_t &other)
{
    // Fake FP8_E5M2 to FP32 convertor
    float res = 0;
    return res;
}

//! FP8_E5M2 precision for StarPU buffers
using fp8_e5m2_t = DType<::int8_t, 1, DTypeEnum::FP8_E5M2, float,
      float_to_e5m2, e5m2_to_float>;

//! Large enough signed integer for indexing purposes
using Index = ::int64_t;

//! Templated printing for DType
template<typename T, size_t nbytes, enum DTypeEnum dtype_enum_,
    typename T_compat, T compat2T(const T_compat &)=T2Y<T_compat, T>,
    T_compat T2compat(const T &)=T2Y<T, T_compat>>
static std::ostream &operator<<(std::ostream &os, const
        DType<T, nbytes, dtype_enum_, T_compat, compat2T, T2compat> &val)
{
    os << val.get();
    return os;
}

} // namespace nntile
