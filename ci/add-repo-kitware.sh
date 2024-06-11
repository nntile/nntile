#!/bin/sh

set -xe

apt update
apt install -y --no-install-recommends \
    ca-certificates curl gpg lsb-release

. /etc/os-release
if [ "$UBUNTU_CODENAME" != "jammy" ]; then
    exit 0
fi

(curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
| gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg)

echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" > /etc/apt/sources.list.d/kitware.list
apt update
apt-get install -y kitware-archive-keyring
