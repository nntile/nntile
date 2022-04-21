#pragma once

#include <bitset>

namespace nntile
{

using fp64_t = double;
using fp32_t = float;

class tf32_t
{
    // 32-bit
    std::bitset<32> value;
public:
    //! Constructor
    constexpr tf32_t(double value_)
    {
        static_assert("Not implemented");
    }
};

class fp16_t
{
    std::bitset<16> value;
public:
    //! Constructor
    constexpr fp16_t(double value_)
    {
        static_assert("Not implemented");
    }
};

class bf16_t
{
    std::bitset<16> value;
public:
    //! Constructor
    constexpr bf16_t(double value_)
    {
        static_assert("Not implemented");
    }
};

} // namespace nntile

