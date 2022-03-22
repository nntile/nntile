#pragma once

#include <complex>

namespace nntile
{

struct BaseType
{
    enum Value: int
    {
        Single = 1,
        Double = 2,
        SingleComplex = 3,
        DoubleComplex = 4
    } value;
    BaseType(const float *):
        value(Single)
    {
    }
    BaseType(const double *):
        value(Double)
    {
    }
    BaseType(const std::complex<float> *):
        value(SingleComplex)
    {
    }
    BaseType(const std::complex<double> *):
        value(DoubleComplex)
    {
    }
    template<typename T>
    explicit operator T() = delete;
    size_t size()
    {
        switch(value)
        {
            case Single:
                return sizeof(float);
                break;
            case Double:
                return sizeof(double);
                break;
            case SingleComplex:
                return sizeof(std::complex<float>);
                break;
            case DoubleComplex:
                return sizeof(std::complex<double>);
                break;
            default:
                throw std::runtime_error("Wrong enum value");
        }
    }
};

} // namespace nntile

