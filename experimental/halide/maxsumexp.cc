#include <iostream>

#include <Halide.h>

#include "maxsumexp.h"

namespace Runtime = Halide::Runtime;

int main(int argc, char *argv[]) {
    Runtime::Buffer<float> input(64, 64, 64), output(64, 64, 2);
    if (int retcode = maxsumexp(input, output); retcode) {
        std::cerr << "failed to execute halide routine: " << retcode << '\n';
        return 1;
    }
    return 0;
}
