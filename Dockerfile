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
    wget \
    curl \
    unzip \
    git \
    git-lfs \
    python2-dev \
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
    tmux

RUN mkdir -p /root/code

# Copy tiny-differentiable-simulator folder
COPY . /root/code/tiny-differentiable-simulator

# Get python dependencies
WORKDIR /root/code/tiny-differentiable-simulator
RUN pip3 install -r python/requirements.txt

# Check out git submodules
WORKDIR /root/code/tiny-differentiable-simulator
RUN git submodule update --init --recursive

#  Build TDS
WORKDIR /root/code/tiny-differentiable-simulator
RUN rm -rf build && cmake -Bbuild . && make -C build experiment_neural_swimmer

# Install Python requirements
WORKDIR /root/code/tiny-differentiable-simulator/python/plotting
RUN pip3 install -r requirements.txt

# Set up git lfs
RUN git lfs install

# Setup repo
WORKDIR /root/code/tiny-differentiable-simulator
