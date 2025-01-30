#!/bin/bash
mkdir -p checkpoints
cd checkpoints
echo "Downloading checkpoints"
wget https://mobility.iiit.ac.in/IDDPed_checkpoints/bitrap.zip
wget https://mobility.iiit.ac.in/IDDPed_checkpoints/intention.zip
wget https://mobility.iiit.ac.in/IDDPed_checkpoints/mtn.zip
wget https://mobility.iiit.ac.in/IDDPed_checkpoints/piefull.zip
wget https://mobility.iiit.ac.in/IDDPed_checkpoints/sgnet.zip

unzip -t bitrap.zip
unzip -t mtn.zip
unzip -t piefull.zip
unzip -t sgnet.zip
unzip -t intention.zip
