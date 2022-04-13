#pragma once

#include <stdexcept>

namespace nntile
{

class TransOp
{
public:
    enum Value: int
    {
        NoTrans,
        Trans
    } value;
    constexpr TransOp(const enum TransOp::Value &value_):
        value(value_)
    {
        switch(value)
        {
            case TransOp::NoTrans:
            case TransOp::Trans:
                break;
            default:
                throw std::runtime_error("Invalid value of TransOp object");
        }
    }
    template<typename T>
    TransOp(const T &) = delete;
    template<typename T>
    operator T() = delete;
};

} // namespace nntile

