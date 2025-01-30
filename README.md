# IDDPeD: Pedestrian Intention and Trajectory Prediction in Unstructured Traffic Environments
This repo contains the implementation of the baselines trained and evaluated on our IDD-PeD dataset.

![image](https://github.com/user-attachments/assets/967755a1-7f4a-4d7d-945f-a845ff188254)

Our dataset provides five types of annotations for pedestrians requiring the ego-vehicle’s attention -
i) Spatial Annotations, which includes tracked bounding boxes for pedestrians.
ii) Behavior Annotations, which are frame-level annotations that capture pedestrian behavior in unstructured environments.
iii) Scene Annotations, capturing the environmental contextual information around the pedestrian.
iv) Interaction Annotations, capturing how pedestrians and vehicles interact in rule-flexible unstructured environments.
v) Location Annotations, providing location context for pedestrian movements.

![Jaywalking is a common occurence in our dataset, given the unstructured traffic environment](./jaywalking_iddp_2.gif)

## Cloning the repo
1. Clone the IDD-PeD repo and navigate to the root directory:
```bash
git clone https://github.com/Ruthvik9/DPJI.git
cd DPJI
```

## Dataset download

2. Run the following command to download the videos of IDD-PeD:
```bash
bash download_videos.sh
```

3. Several methods require individual video frames for training and testing. Run the following code as a python script to extract frames from the videos:

```python
from iddped_interface_traj import IDDPedestrian
dataset = IDDPedestrian()
dataset.extract_and_save_images()
```
4. This will extract and save frames as .png images in the `data/IDDPedestrian/images` directory

## Intention Prediction

### Setup
1. Create the conda environment using the provided configuration file and navigate to the directory:
```bash
conda env create -f envs/intention_config.yml
conda activate baseline
cd Intention
```

### Testing
2. To re-run test on the saved model use:
```bash
python test_model.py <saved_files_path>
```
For example,
```bash
python test_model.py models/jaad/MASK_PCPA/xxxx/
```


### Training and Testing
3. Run the following script to train and test the intention prediction model:
```bash
bash run_all_on_iddped.sh
```

## Trajectory Prediction

### PIEPredict
1. Setup environment and navigate to the directory:
```bash
conda env create -f envs/PIEPredict_config.yml
conda activate traj
cd PIEPredict
```

### Training and testing
To train all models from scratch and evaluate them on the test data use this command:
```bash
python train_test.py 1
```
This will train intention, speed and trajectory models separately and evaluate them on the test data.

_Note: training intention model uses image data and requires 32GB RAM.

### Testing:
```bash
python train_test.py 2
```

### MTN (Multiple Trajectory Network)
1. Setup environment and navigate to the directory:
```bash
conda env create -f envs/MTN_config.yml
conda activate MTN
cd MTN
```

2. Run prediction:
```bash
python train.py
```

### BiTraP
1. Setup environment and navigate to the directory:
```bash
conda env create -f envs/bitrap_config.yml
conda activate bitrap
cd BITRAP
```

2. Testing:
```bash
python tools/test.py --config_file configs/bitrap_np_IDDP.yml CKPT_DIR epoch_latest.pth
```

3. Training and testing:
```bash
python tools/train.py --config_file configs/bitrap_np_IDDP.yml CKPT_DIR epoch_latest.pth
```

### SGNet
1. Setup environment and navigate to the directory:
```bash
conda env create -f envs/SGNet_config.yml
conda activate SGNet
cd SGNet
```

### Training
2. Run prediction:
```bash

```


## Citation
If you find this work useful, please cite our paper:
```bibtex
[Citation placeholder]
```
