# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/CMakeLists.txt
# Python wrappers for High-level API of NNTile library
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-12-18

pybind11_add_module(nntile_core "nntile/nntile_core.cc")
target_link_libraries(nntile_core PRIVATE nntile)

get_target_property(NNTILE_CORE_POSTFIX nntile_core DEBUG_POSTFIX)
get_target_property(NNTILE_CORE_SUFFIX nntile_core SUFFIX)
set(NNTILE_CORE_NAME
    "nntile_core${NNTILE_CORE_DEBUG_POSTFIX}${NNTILE_CORE_SUFFIX}")
set(NNTILE_CORE_SYMLINK ${NNTILE_CORE_SYMLINK_DIR}/${NNTILE_CORE_NAME})

add_custom_command(
    OUTPUT ${NNTILE_CORE_SYMLINK}
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        $<TARGET_FILE:nntile_core> ${NNTILE_CORE_SYMLINK}
    DEPENDS nntile_core)

add_custom_target(nntile-symlink DEPENDS ${NNTILE_CORE_SYMLINK})
