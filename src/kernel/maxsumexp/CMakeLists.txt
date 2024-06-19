# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.

add_custom_target(nntile.kernel.bench-all)

add_executable(nntile.kernel.maxsumexp-bench maxsumexp_bench.cu)
target_link_libraries(nntile.kernel.maxsumexp-bench
    PRIVATE nntile nvbench::main)
