#include "nntile/layer/fully_connected.hh"
#include "../testing.hh"
#include <iostream>

using namespace nntile;

template<typename T>
void validate_fc()
{
    FullyConnected<T> layer1({10, 20, 30}, {3, 4, 5});
    Tensor<T> input1({30, 20}, {5, 4}), output1({10, 20, 20}, {3, 4, 4});
    layer1.forward_async(input1, output1);
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_fc<fp32_t>();
    validate_fc<fp64_t>();
    return 0;
}

