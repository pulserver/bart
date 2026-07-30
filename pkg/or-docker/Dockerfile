# README
# We need:
# - BART's .git-folder in the current directory
# - ISMRMRD's .git-folder at ismrmrd/.git
# - a folder names "cuda-11-6-2" containing
#   - the cuda .deb local installer available at
#     https://developer.nvidia.com/cuda-11-6-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_local
#   - liburcu6_0.12.2-1_amd64.deb, a cuda dependency available at
#     ftp.at.debian.org/debian/pool/main/libu/liburcu/liburcu6_0.12.2-1_amd64.deb
# - BART and ISMRMRD paths can be customized via docker buildtime args below; for CUDA it's not possible as variables can not be used in bind-mounts on debian 13 podman

ARG BART=.git
ARG ISMRMRD=ismrmrd/.git


ARG DEBUG_HOST=
ARG OR_GPU_SUPPORT=1


FROM debian:13 AS build
ARG OR_GPU_SUPPORT
ARG BART
ARG ISMRMRD
SHELL ["/bin/bash", "-c"]

# Build dependencies
RUN apt-get update                                                                                  && \
    apt-get install -y git                                                                          \
    cmake gcc g++ libhdf5-dev libxml2-dev libxslt1-dev libboost-all-dev libfftw3-dev libpugixml-dev \
    make git libfftw3-dev liblapacke-dev libpng-dev

# cuda dependencies: gcc-11
RUN echo deb http://ftp.de.debian.org/debian bookworm main > /etc/apt/sources.list.d/bookworm.list
RUN apt-get update  && apt-get install -y gcc-11 g++-11


# ISMRMRD
COPY --from=deps $ISMRMRD ismrmrd-git
RUN <<EOF
set -eu
mkdir -p build
git clone file:///ismrmrd-git build/ismrmrd
cd /build/ismrmrd
mkdir -p build
cd build
cmake ../
make -j $(nproc)
make install
EOF


# CUDA for GPU-accelerated BART
## cuda repo
RUN --mount=type=bind,dst=/cuda,source=cuda-11-6-2,from=deps <<EOF
set -eu
if [ "$OR_GPU_SUPPORT" = "1" ]; then
    dpkg -i /cuda/cuda-repo-debian11-11-6-local_11.6.2-510.47.03-1_amd64.deb
    dpkg -i /cuda/liburcu6_0.12.2-1_amd64.deb
    cp /var/cuda-repo-debian11-11-6-local/7fa2af80.pub /etc/apt/trusted.gpg.d/cuda.asc
    apt-get update
fi
EOF

## cuda itself
RUN --mount=type=bind,dst=/cuda,source=cuda-11-6-2,from=deps \
 if [ "$OR_GPU_SUPPORT" = "1" ]; then\
 apt-get install -y cuda-command-line-tools-11-6 cuda-libraries-dev-11-6;\
 fi

## patch headers for glibc exception specification change in deb 13
RUN --mount=type=bind,dst=/cuda,source=cuda-11-6-2,from=deps \
 if [ "$OR_GPU_SUPPORT" = "1" ]; then \
 cd /usr/local/cuda//bin/../targets/x86_64-linux/include/crt/ && patch < /cuda/cuda-11-patch-all;\
 fi

ENV PATH=/usr/local/cuda/bin:${PATH}


# BART
COPY $BART bart-git
RUN <<EOF
set -eu
mkdir -p build
git clone file:///bart-git build/bart
cd /build/bart
make -j $(nproc)                                                                    \
    CUDA=${OR_GPU_SUPPORT} CUDA_BASE=/usr/local/cuda/ CUDA_LIB=lib64                \
    ISMRMRD=1 ISMRM_H="-I/usr/local/include/ismrmrd -I/usr/include/hdf5/serial/"    \
    CUDA_CC=gcc-11
make install
EOF

##############################################################

FROM debian:13
ARG OR_GPU_SUPPORT
ARG DEBUG_HOST
SHELL ["/bin/bash", "-c"]

# CUDA runtime
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN --mount=type=bind,dst=/cuda,source=cuda-11-6-2,from=deps <<EOF
set -eu
if [ "$OR_GPU_SUPPORT" = "1" ]; then
    dpkg -i /cuda/cuda-repo-debian11-11-6-local_11.6.2-510.47.03-1_amd64.deb
    dpkg -i /cuda/liburcu6_0.12.2-1_amd64.deb
    cp /var/cuda-repo-debian11-11-6-local/7fa2af80.pub /etc/apt/trusted.gpg.d/cuda.asc
    apt-get update
    apt-get install -y --no-install-recommends cuda-cudart-11-6 libcublas-11-6 libcufft-11-6
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    dpkg -r cuda-repo-debian11-11-6-local
    rm /etc/apt/sources.list.d/cuda-debian11-11-6-local.list
fi
EOF

# Runtime dependencies and utilities
RUN <<EOF
set -eu
apt-get update
apt-get install -y libboost-program-options1.83.0 libgcc-s1 libhdf5-310 libpugixml1v5   \
    libfftw3-single3                                                                    \
    liblapacke libpng16-16t64 libblas3 libgomp1 \
    ncat bc vim-tiny openssh-client
if [ -n "${DEBUG_HOST:-}" ]; then
    apt-get install -y openssh-server
fi
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF


# ISMRMRD library
COPY --from=build /usr/local/lib/libismrmrd.so.*  /usr/local/lib/
ENV LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

# BART binaries
COPY --from=build /usr/local/bin/bart             /usr/local/bin/
COPY --from=build /usr/local/lib/bart/commands    /usr/local/lib/bart/commands
COPY --from=build /usr/local/share/doc/bart       /usr/local/share/doc/bart

# BART scripts
COPY scripts /bart/scripts
RUN ln -s /usr/local/bin/bart /bart/bart
ENV BART_TOOLBOX_PATH=/bart
ENV OR_GPU_SUPPORT=$OR_GPU_SUPPORT


# Debugging shell
RUN passwd -d root
ENV DEBUG_HOST=$DEBUG_HOST

CMD ["/bin/bash", "-c", "/usr/sbin/ldconfig && $BART_TOOLBOX_PATH/scripts/or-server.sh"]
