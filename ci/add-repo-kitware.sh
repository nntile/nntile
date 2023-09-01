#!/bin/sh

set -xe

apt install -y --no-install-recommends \
    ca-certificates curl gpg

(curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
| gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg)

. /etc/os-release
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" > /etc/apt/sources.list.d/kitware.list
