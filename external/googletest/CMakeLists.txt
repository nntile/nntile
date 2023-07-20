set(BUILD_GMOCK OFF)
set(INSTALL_GTEST OFF)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
    URL_HASH SHA256=ad7fdba11ea011c1d925b3289cf4af2c66a352e18d4c7264392fead75e919363
    FIND_PACKAGE_ARGS GLOBAL NAMES GTest
)
