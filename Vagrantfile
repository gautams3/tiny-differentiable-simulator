# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/bionic64"
  config.vm.network "private_network", ip: "192.168.121.150"

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
  python-matplotlib \
  python-numpy
SHELL

end
