#!/bin/sh
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file ci/add-repo-kitware
# A script to add source of CMake package to the APT system
#
# @version 1.1.0

set -xe

apt update
apt install -y --no-install-recommends \
    ca-certificates curl gpg lsb-release

. /etc/os-release

(curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
| gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg)

echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" > /etc/apt/sources.list.d/kitware.list
apt update
apt-get install -y kitware-archive-keyring
