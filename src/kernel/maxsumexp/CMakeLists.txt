add_custom_target(nntile.kernel.bench-all)

add_executable(nntile.kernel.maxsumexp-bench maxsumexp_bench.cu)
target_link_libraries(nntile.kernel.maxsumexp-bench
    PRIVATE nntile nvbench::main)
