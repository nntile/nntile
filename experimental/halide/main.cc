#include <iostream>
#include <vector>

#include <Halide.h>

using Halide::Buffer, Halide::Expr, Halide::Func, Halide::Var, Halide::Tuple;

class MaxSumExpGenerator : public Halide::Generator<MaxSumExpGenerator> {
public:
    Input<Buffer<float, 1>> input{"input"};

    Output<Buffer<float, 1>> output{"output"};

    Var i;

    void generate() {
        output(i) = input(i);
    }
};

HALIDE_REGISTER_GENERATOR(MaxSumExpGenerator, maxsumexp);

int main(int argc, char *argv[]) {
    constexpr auto size = 4;
    std::vector<int> shape = {size, size, size};
    std::vector<float> data(size * size * size);
    for (auto it = 0; it != size * size * size; ++it) {
        data[it] = it;
    }
    // Buffer<float, 3> input(data.data(), shape, "input");
    Halide::ImageParam input(Halide::type_of<float>(), 3);

    Var i, j, k;

    Func state;
    // state.trace_stores();
    state(i, j) = {input(i, j, 0), Expr(1.0f)};

    Halide::RDom r(1, size - 1);
     Expr prev_max = state(i, j)[0];
     Expr prev_sum = state(i, j)[1];
     Tuple keep_max = {
         prev_max,
         prev_sum + Halide::exp(input(i, j, r) - prev_max)};
     Tuple update_max = {
         input(i, j, r),
         prev_sum * Halide::exp(prev_max - input(i, j, r)) + Expr(1)};
     state(i, j) = Halide::select(
         prev_max >= input(i, j, r), keep_max, update_max);

    Func maxsumexp;
    maxsumexp(i, j, k) = Halide::select(k == 0, state(i, j)[0], state(i, j)[1]);
    // maxsumexp.compile_to_lowered_stmt("maxsumexp.stmt.html", {}, Halide::HTML);

    maxsumexp.compile_to_static_library("maxsumexp", {input}, "maxsumexp");

    {
        // maxsumexp.realize({size, size, 2});
        // Halide::Realization result = maxsumexp.realize({size});
        // Buffer<float> max = result[0];
        // Buffer<float> sum = result[1];
        // std::cout << "Max: " << max(0) << std::endl;
        // std::cout << "Sum: " << sum(0) << std::endl;
    }

    return 0;
}
