/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/constants.cc
 * Test TransOp
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-19
 * */

#include "nntile/constants.hh"
#include "testing.hh"

using namespace nntile;

int main(int argc, char **argv)
{
    {volatile TransOp x(TransOp::NoTrans);};
    {volatile TransOp x(TransOp::Trans);};
    for(int i = -10; i < 10; ++i)
    {
        auto value = static_cast<enum TransOp::Value>(i);
        switch(value)
        {
            case TransOp::NoTrans:
            case TransOp::Trans:
                {volatile TransOp x(value);};
                break;
            default:
                TEST_THROW(volatile TransOp x(value));
        }
    }
    return 0;
}

