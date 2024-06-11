set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_VERSION 2.12.0)

FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v${PYBIND11_VERSION}.tar.gz
    URL_HASH SHA256=bf8f242abd1abcd375d516a7067490fb71abd79519a282d22b6e4d19282185a7
    FIND_PACKAGE_ARGS GLOBAL NAMES pybind11
)
