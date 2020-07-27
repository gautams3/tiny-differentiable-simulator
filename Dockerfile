FROM ubuntu:18.04

ENV LANG C.UTF-8

RUN \
  apt-get -y -q update && \
  # Prevents debconf from prompting for user input
  # See https://github.com/phusion/baseimage-docker/issues/58
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gcc \
    g++ \
    build-essential \
    cmake \
    wget \
    curl \
    unzip \
    git \
    git-lfs \
    python-dev \
    python-pip \
    python3-dev \
    python3-pip \
    libeigen3-dev \
    libceres-dev \
    libtinyxml2-dev \
    libicu-dev \
    libbz2-dev \
    freeglut3-dev \
    ffmpeg \
    libhdf5-dev \
    nano \
    htop \
    tmux \
    libcrossguid-dev


# apt install python-pip2
# apt install python-pip
# apt install cmake
# apt install uuid-dev
# apt-get install libhdf5-dev
# apt search crossguid
# apt install libcrossguid-dev
# apt search boost
# apt install libboost-dev
# apt install libboost-system
# apt search boost-system
# apt install libboost-system-dev
# apt install libboost-filesystem-dev
# apt install libboost-serialization-dev
# apt search gcc
# apt install g++
# apt install gcc-9
# apt install gcc-8
# apt-get install manpages-dev
# apt install software-properties-common
# add-apt-repository ppa:ubuntu-toolchain-r/test
# apt install gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9
# apt install clang-9
# apt search numpy
# apt install python-numpy
# apt install -y cgdb
# apt install -y vim-nox
# apt install valgrind

RUN mkdir -p /root/code

# Copy tiny-differentiable-simulator folder
COPY . /root/code/tiny-differentiable-simulator

# Get python dependencies
WORKDIR /root/code/tiny-differentiable-simulator
RUN pip3 install -r python/requirements.txt

# Check out git submodules
# WORKDIR /root/code/tiny-differentiable-simulator
# RUN git submodule update --init --recursive

# #  Build TDS
# WORKDIR /root/code/tiny-differentiable-simulator
# RUN rm -rf build && cmake -Bbuild . && make -C build experiment_neural_swimmer

# Set up git lfs
RUN git lfs install

# Setup repo
WORKDIR /root/code/tiny-differentiable-simulator
