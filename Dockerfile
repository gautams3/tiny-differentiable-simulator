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
    autotools-dev \
    m4 \
    libicu-dev \
    build-essential \
    libbz2-dev \
    libasio-dev \
    libeigen3-dev \
    freeglut3-dev \
    expat \
    libcairo2-dev \
    cmake \
    python3-pip \
    ffmpeg \
    libhdf5-dev \
    nano \
    htop \
    tmux


RUN mkdir -p /root/code

# Copy tiny-differentiable-simulator folder
COPY . /root/code/tiny-differentiable-simulator
WORKDIR /root/code/tiny-differentiable-simulator

RUN pip3 install -r python/requirements.txt

# Install ipywidgets for Jupyter Lab
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Build and install libccd 1.4
WORKDIR /root/code
RUN wget https://github.com/danfis/libccd/archive/v1.4.zip && unzip v1.4.zip && cd libccd-1.4/src && echo "#define CCD_FLOAT" | cat - ccd/vec3.h > /tmp/out && mv /tmp/out ccd/vec3.h && make -j4 && make install

# Check out git submodules
WORKDIR /root/code/tiny-differentiable-simulator
RUN git submodule init && git submodule update

# Build and install SBPL
WORKDIR /root/code
RUN git clone https://github.com/sbpl/sbpl.git && cd sbpl && mkdir build && cd build && cmake .. && make -j4 && make install

# Build and install OMPL
WORKDIR /root/code/tiny-differentiable-simulator
RUN cd ompl && mkdir build && cd build && cmake .. && make -j4 && make install

# Creating Build Files
WORKDIR /root/code/tiny-differentiable-simulator
RUN rm -rf build && cmake -H. -Bbuild

# Build tiny-differentiable-simulator
RUN cd build && make

# Run benchmark executable to generate benchmark_template.json
WORKDIR /root/code/tiny-differentiable-simulator/bin
RUN benchmark

# Install Python requirements for plotting
WORKDIR /root/code/tiny-differentiable-simulator/plotting
RUN pip3 install -r requirements.txt

# Set up git lfs
RUN git lfs install

# Setup repo
WORKDIR /root/code/tiny-differentiable-simulator

# Use bash as default shell
SHELL ["/bin/bash", "-c"]

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]