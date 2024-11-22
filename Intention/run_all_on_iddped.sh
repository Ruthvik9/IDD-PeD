#!/bin/bash

# benchmark comparison
python train_test.py -c config_files_iddped/baseline/Static_vgg16.yaml # Static
python train_test.py -c config_files_iddped/baseline/ConvLSTM_vgg16.yaml # Conv LSTM
python train_test.py -c config_files_iddped/baseline/C3D.yaml # C3D
python train_test.py -c config_files_iddped/baseline/I3D.yaml # I3D
python train_test.py -c config_files_iddped/baseline/PCPA_jaad.yaml  # PCPA
python train_test.py -c config_files_iddped/baseline/SingleRNN.yaml  # SingleRNN
python train_test.py -c config_files_iddped/baseline/SFRNN.yaml      # SF-GRU
python train_test.py -c config_files_iddped/ours/MASK_PCPA_jaad_2d.yaml   # MaskPCPA
