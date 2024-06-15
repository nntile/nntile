# NNTile: Manylinux Wheels

## Overview

In order to produce distributable wheels (so called manylinux-wheel). In order
to build wheel, one should run a building script which bootstrap a
containerized environment with all dependencies. For example, the following
command builds a wheel for Python 3.12.

```shell
ci/manylinux/build.sh 3.12
```

The base image is `quay.io/pypa/manylinux_2_28_x86_64` which roughly
corresponds to Ubuntu 20.04. In other words, wheel installed on Ubuntu 20.04 or
newer will work.

Another limitation is that we do not support MPI and CUDA at the moment.
