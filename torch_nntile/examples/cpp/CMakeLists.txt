# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/cpp/CMakeLists.txt

add_executable(torch_nntile_train_deep_relu train_deep_relu.cpp)
target_link_libraries(torch_nntile_train_deep_relu PRIVATE torch_nntile)
