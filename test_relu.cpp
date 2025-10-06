#include <iostream>
#include <vector>
#include "nntile/kernel/relu/cpu.hh"

int main() {
    std::vector<float> data = {-1.0f, 0.0f, 1.0f};
    nntile::kernel::relu::cpu<float>(3, data.data());
    for(float x : data) std::cout << x << ' ';
    std::cout << std::endl;
    return 0;
}
