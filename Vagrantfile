# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/bionic64"
  config.vm.network "private_network", ip: "192.168.121.150"

  config.vm.provider "virtualbox" do |v|
    v.memory = 4096
    v.cpus = 2
  end

  config.vm.provision "shell", inline: <<-SHELL
sudo rm /etc/localtime
sudo ln -s /usr/share/zoneinfo/America/New_York /etc/localtime

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  libceres-dev \
  libpython2.7-dev \
  libpython3.8-dev \
  libtinyxml2-dev \
  python-matplotlib \
  python-numpy \
  libhdf5-dev

if ! [[ -d cxxopts ]]; then
  git clone https://github.com/jarro2783/cxxopts.git
fi
pushd cxxopts
mkdir -p build
pushd build
cmake ..
sudo make install
popd
popd

if ! [[ -d HighFive ]]; then
  git clone https://github.com/BlueBrain/HighFive.git
fi
pushd HighFive
mkdir -p build
pushd build
cmake -DHIGHFIVE_USE_BOOST=OFF ..
sudo make install
popd
popd

if ! [[ -d bullet3 ]]; then
  git clone https://github.com/bulletphysics/bullet3.git
fi
pushd bullet3
./build_cmake_pybullet_double.sh
pushd build_cmake
sudo make install
popd
popd

SHELL

end
