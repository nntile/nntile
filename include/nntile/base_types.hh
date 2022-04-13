#pragma once

#include <cstddef>

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

} // namespace nntile

