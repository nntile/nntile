#include "nntile/layer/fully_connected.hh"
#include "../testing.hh"
#include <iostream>

using namespace nntile;
Starpu starpu;

template<typename T>
void validate_fc()
{
    FullyConnected<T> layer1({10, 20, 30}, {3, 4, 5});
    unsigned long long seed = -1;
    T mean = 0, stddev = 1;
    layer1.init(seed, mean, stddev);
    Tensor<T> input1({30, 20}, {5, 4}), output1({10, 20, 20}, {3, 4, 4});
    randn(input1, seed-1, mean, stddev);
    layer1.forward_async(input1, output1);
    starpu.wait_for_all();
}

int main(int argc, char **argv)
{
    validate_fc<fp32_t>();
    validate_fc<fp64_t>();
    return 0;
}

