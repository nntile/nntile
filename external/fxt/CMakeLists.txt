if (SYSTEM_FXT)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FxT REQUIRED IMPORTED_TARGET GLOBAL fxt)
    add_library(fxt::fxt ALIAS PkgConfig::FxT)
else()
    find_program(AUTORECONF_BINARY NAMES autoreconf REQUIRED)
    find_program(MAKE_BINARY NAMES make REQUIRED)

    set(BUILD_COMMAND "${MAKE_BINARY}")
    if (DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
        set(APPEND BUILD_COMMAND "-j${CMAKE_BUILD_PARALLEL_LEVEL}")
    endif()

    FetchContent_Declare(fxt
        URL https://download.savannah.nongnu.org/releases/fkt/fxt-0.3.13.tar.gz
        URL_HASH SHA256=1ad925b3df678c47321524ca5f2f811cdd3edcb19706b7e2a07c394c63e62eff
        PATCH_COMMAND patch -p1 -i${CMAKE_CURRENT_SOURCE_DIR}/fxt.diff
    )

    FetchContent_Populate(fxt)

    include(ExternalProject)

    ExternalProject_Add(fxt
        SOURCE_DIR ${fxt_SOURCE_DIR}
        BINARY_DIR ${fxt_BINARY_DIR}
        CONFIGURE_HANDLED_BY_BUILD TRUE
        CONFIGURE_COMMAND /bin/sh -c "
            ${CMAKE_COMMAND} -E copy_directory ${fxt_SOURCE_DIR} ${fxt_BINARY_DIR} &&
            ${AUTORECONF_BINARY} &&
            ./configure --prefix=/usr --disable-shared --enable-languages=c,c++ --enable-static --with-pic &&
            ${CMAKE_COMMAND} -E make_directory fxt &&
            ${CMAKE_COMMAND} -E copy tools/fut.h tools/fxt.h tools/fxt-tools.h fxt"
        BUILD_COMMAND "${BUILD_COMMAND}"
        BUILD_BYPRODUCTS
            "${fxt_BINARY_DIR}/fxt/fxt.h"
            "${fxt_BINARY_DIR}/tools/.libs/libfxt.a"
        LOG_CONFIGURE ON
        LOG_BUILD ON
    )

    ExternalProject_Add_StepTargets(fxt build configure)

    # NOTE Not sure about PARENT_SCOPE but this simplifies a bit building StarPU.
    set(FXT_LINK_LIBRARIES "fxt" PARENT_SCOPE)
    set(FXT_LINK_OPTIONS "-L${fxt_BINARY_DIR}/tools/.libs" PARENT_SCOPE)
    set(FXT_INCLUDE_DIRECTORIES "${fxt_BINARY_DIR}" PARENT_SCOPE)

    add_library(fxt-static STATIC IMPORTED GLOBAL)
    add_library(fxt::fxt ALIAS fxt-static)
    add_dependencies(fxt-static fxt-build)
    set_target_properties(fxt-static PROPERTIES
        IMPORTED_LOCATION "${fxt_BINARY_DIR}/tools/.libs/libfxt.a"
        INTERFACE_INCLUDE_DIRECTORIES "${fxt_BINARY_DIR}")
endif()
