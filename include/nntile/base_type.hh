#pragma once

#include <cstddef>
#include <stdexcept>

namespace nntile
{

using fp64_t = double;
using fp32_t = float;

class tf32_t
{
    int32_t value;
public:
    //! Constructor
    constexpr tf32_t(float value_):
        value(*reinterpret_cast<int32_t *>(&value_))
    {
    }
};

class fp16_t
{
    int16_t value;
public:
    //! Constructor
    constexpr fp16_t(float value_):
        value(*reinterpret_cast<int16_t *>(&value_))
    {
    }
};

class bf16_t
{
    int16_t value;
public:
    //! Constructor
    constexpr bf16_t(float value_):
        value(*reinterpret_cast<int16_t *>(&value_))
    {
    }
};

struct BaseType
{
    enum Value: int
    {
        Undefined = 0,
        FP64 = 1,
        FP32 = 2,
        TF32 = 3,
        FP16 = 4,
        BF16 = 5,
    } value;
    explicit constexpr BaseType(const enum Value &value_):
        value(value_)
    {
    }
    explicit constexpr BaseType(fp64_t):
        value(BaseType::FP64)
    {
    }
    explicit constexpr BaseType(fp32_t):
        value(BaseType::FP32)
    {
    }
    explicit constexpr BaseType(tf32_t):
        value(BaseType::TF32)
    {
    }
    explicit constexpr BaseType(fp16_t):
        value(BaseType::FP16)
    {
    }
    explicit constexpr BaseType(bf16_t):
        value(BaseType::BF16)
    {
    }
    template<typename T>
    explicit operator T() = delete;
    constexpr size_t size() const
    {
        switch(value)
        {
            case FP64:
                return 8;
                break;
            case FP32:
            case TF32:
                return 4;
                break;
            case FP16:
            case BF16:
                return 2;
                break;
            default:
                throw std::runtime_error("Wrong enum value");
        }
    }
};

} // namespace nntile

