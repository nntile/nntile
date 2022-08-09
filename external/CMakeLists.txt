# @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file external/CMakeLists.txt
# External header libraires to be used
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2022-08-09

# Set list of external sources and headers
set(EXTERNAL
    "random.h"
    )

target_sources(nntile PRIVATE ${EXTERNAL})

