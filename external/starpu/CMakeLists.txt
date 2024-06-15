if (SYSTEM_STARPU)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(StarPU REQUIRED IMPORTED_TARGET GLOBAL starpu-1.3)
    add_library(starpu::starpu ALIAS PkgConfig::StarPU)
else()
    find_program(AUTORECONF_BINARY NAMES autoreconf REQUIRED)
    find_program(MAKE_BINARY NAMES gmake nmake make REQUIRED)

    # NOTE For some reason, CMake does not set CMAKE_BUILD_PARALLEL_LEVEL
    # variable with -j or --parallel options. Thus we set number of jobs here
    # to 8.
    set(BUILD_COMMAND "${MAKE_BINARY}")
    if (DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
        set(APPEND BUILD_COMMAND "-j${CMAKE_BUILD_PARALLEL_LEVEL}")
    else()
        set(APPEND BUILD_COMMAND "-j8")
    endif()

    FetchContent_Declare(starpu
        URL https://github.com/starpu-runtime/starpu/archive/refs/tags/starpu-1.3.11.tar.gz
        URL_HASH SHA256=43b27c48a37722ba0e596684ba2e723d72b0f6b020a12c25843f8306049a5af0
    )

    FetchContent_Populate(starpu)

    file(MAKE_DIRECTORY "${starpu_BINARY_DIR}/include")
    list(TRANSFORM FXT_LINK_LIBRARIES PREPEND "-l" OUTPUT_VARIABLE FXT_LIBS)

    include(ExternalProject)

    ExternalProject_Add(starpu
        DEPENDS fxt-build
        SOURCE_DIR ${starpu_SOURCE_DIR}
        BINARY_DIR ${starpu_BINARY_DIR}
        CONFIGURE_HANDLED_BY_BUILD TRUE
        CONFIGURE_COMMAND /bin/sh -c "
            ${CMAKE_COMMAND} -E copy_directory_if_different ${starpu_SOURCE_DIR} ${starpu_BINARY_DIR} &&
            ${AUTORECONF_BINARY} -ivf -I m4 &&
            export FXT_CFLAGS=-I${FXT_INCLUDE_DIRECTORIES} && \
            export FXT_LDFLAGS=${FXT_LINK_OPTIONS} && \
            export FXT_LIBS=${FXT_LIBS} && \
            ./configure \
                --disable-build-doc \
                --disable-build-doc-pdf \
                --disable-build-examples \
                --disable-build-tests \
                --disable-cuda \
                --disable-fortran \
                --disable-mpi \
                --disable-opencl \
                --disable-socl \
                --disable-starpufft \
                --disable-starpupy \
                --disable-static \
                --enable-blas-lib=none \
                --enable-cpu \
                --enable-maxbuffers=16 \
                --enable-shared \
                --prefix=/usr \
                --with-fxt \
                --with-pic"
        BUILD_COMMAND "${BUILD_COMMAND}"
        BUILD_BYPRODUCTS
            "${starpu_BINARY_DIR}/include"
            "${starpu_BINARY_DIR}/include/starpu_config.h"
            "${starpu_BINARY_DIR}/src/.libs/libstarpu-1.3.so"
        LOG_CONFIGURE ON
        LOG_BUILD ON
    )

    ExternalProject_Add_StepTargets(starpu build configure)

    add_library(starpu-shared SHARED IMPORTED GLOBAL)
    add_library(starpu::starpu ALIAS starpu-shared)
    add_dependencies(starpu-shared starpu-build)
    set_target_properties(starpu-shared PROPERTIES
        IMPORTED_LOCATION "${starpu_BINARY_DIR}/src/.libs/libstarpu-1.3.so"
        INTERFACE_INCLUDE_DIRECTORIES "${starpu_BINARY_DIR}/include")
endif()
