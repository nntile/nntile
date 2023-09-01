set(PYBIND11_VERSION 2.11.0)

FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v${PYBIND11_VERSION}.tar.gz
    URL_HASH SHA256=7af30a84c6810e721829c4646e31927af9d8861e085aa5dd37c3c8b8169fcda1
    FIND_PACKAGE_ARGS GLOBAL NAMES pybind11
)
