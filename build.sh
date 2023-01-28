#!/bin/sh
if [ -d "./build" ]
then
    echo "rm old build"
    rm -rf build
fi
echo "rebuild"
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build .
ctest
