#!/bin/bash

# Create the IDDPedestrian directory
mkdir -p IDDPedestrian
cd IDDPedestrian

# Download all the videos
echo "Downloading video tars..."
wget http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0001.tar
wget http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0002.tar
wget http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0003.tar
wget http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0004.tar
wget http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0005.tar
wget http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0006.tar
wget http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0007.tar
wget http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0008.tar
wget http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0009.tar

echo "Untaring the directories..."
tar -xvf gp_set_0001.tar
rm -r gp_set_0001.tar
tar -xvf gp_set_0002.tar
rm -r gp_set_0002.tar
tar -xvf gp_set_0003.tar
rm -r gp_set_0003.tar
tar -xvf gp_set_0004.tar
rm -r gp_set_0004.tar
tar -xvf gp_set_0005.tar
rm -r gp_set_0005.tar
tar -xvf gp_set_0006.tar
rm -r gp_set_0006.tar
tar -xvf gp_set_0007.tar
rm -r gp_set_0007.tar
tar -xvf gp_set_0008.tar
rm -r gp_set_0008.tar
tar -xvf gp_set_0009.tar
rm -r gp_set_0009.tar


# Download the annotations

