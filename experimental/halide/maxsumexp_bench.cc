#include <memory>
#include <random>

#include <Halide.h>
#include <benchmark/benchmark.h>

#include "maxsumexp.h"
#include "nntile/kernel/maxsumexp.hh"

namespace Runtime = Halide::Runtime;

std::unique_ptr<float[]> MakeData(auto &&rng, size_t size) {
    auto ptr = std::make_unique<float[]>(size);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto it = 0; it != size; ++it) {
        ptr[it] = dist(rng);
    }
    return ptr;
}

static void MaxSumExpHalide(benchmark::State& state) {
    std::mt19937_64 rng{42};
    int size = state.range();
    auto input_data = MakeData(rng, size * size * size);
    auto output_data = MakeData(rng, size * size * 2);
    Runtime::Buffer<float> input(input_data.get(), {size, size, size});
    Runtime::Buffer<float> output(output_data.get(), {size, size, 2});

    for (auto _ : state) {
        maxsumexp(input, output);
    }
}

BENCHMARK(MaxSumExpHalide)->Arg(64)->Arg(256)->Arg(512);

static void MaxSumExpNNTile(benchmark::State& state) {
    std::mt19937_64 rng{42};
    int size = state.range();
    auto input = MakeData(rng, size * size * size);
    auto output = MakeData(rng, size * size * 2);
    auto inp = reinterpret_cast<nntile::fp32_t *>(input.get());
    auto out = reinterpret_cast<nntile::fp32_t *>(output.get());

    for (auto _ : state) {
        nntile::kernel::maxsumexp::cpu<nntile::fp32_t>(size, size, size, inp, out);
    }
}

BENCHMARK(MaxSumExpNNTile)->Arg(64)->Arg(256)->Arg(512);

BENCHMARK_MAIN();
