FROM ubuntu:22.04

# Install all the necessary dependencies both for StarPU installation and for
# NNTile development
RUN apt update && apt install -y --no-install-recommends \
    build-essential libopenmpi3 libopenmpi-dev openmpi-bin \
    lsb-release software-properties-common pkg-config \
    fxt-tools libfxt-dev autoconf automake libtool-bin \
    libhwloc-dev vim wget git gdb libopenblas-dev lcov \
    python3 ninja-build binutils clang python3-pip python3-dev

## Add Kitware repo for the latest CMake package
# Get Kitware's signing key
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
    2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg \
    >/dev/null
# Add Kitware's repo to the list of sources
RUN apt-add-repository \
    "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
# Get up-to-date Kitware keyring
#RUN apt install kitware-archive-keyring
# Remove GPG key
#RUN rm /etc/apt/trusted.gpg.d/kitware.gpg
# Copy public key
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4
## Finally, install CMake
RUN apt update && apt install -y --no-install-recommends \
    cmake cmake-curses-gui

# Clear apt cache
RUN apt clean && rm -rf /var/lib/apt/lists/*

# Get the latest StarPU, compile and install it
RUN git clone https://gitlab.inria.fr/starpu/starpu.git /starpu
WORKDIR /starpu
# Use the same version across different hardware
RUN git checkout 51afa9119289d29076b563f264235abfcf7d24ff
RUN ./autogen.sh
RUN ./configure --disable-build-doc --disable-build-tests \
    --disable-build-examples --disable-fortran --disable-opencl
RUN make -j 4 install
# Remove build files of StarPU
RUN rm -rf /starpu

# Set up workspace for NNTile
RUN mkdir -p /workspace
RUN chmod 777 /workspace
WORKDIR /workspace

# Set path to libstarpumpi library
ENV LD_LIBRARY_PATH=/usr/local/lib
# Enable use of MPI with the root user
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_ALLOW_RUN_AS_ROOT=1

# Set entrypoint (disabled for CI/CD)
#ENTRYPOINT ["bash"]

# Install required Python packages
RUN pip install pytest numpy

