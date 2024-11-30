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

## Dataset Setup

1. Place the video files in the `data/IDDPedestrian/videos` folder
2. The videos are present at the `/mnt/base/dpji/IDDPedestrian/videos/` folder on 10.4.16.30.
3. Remove audio from the videos using 'bash remove_audio.sh'
```bash
cd data/IDDPedestrian/
bash remove_audio.sh
```
4. From the root directory, run the following code to extract frames from videos:

```python
from iddped_interface_traj import IDDPedestrian
dataset = IDDPedestrian()
dataset.extract_and_save_images()
```
5. This will extract and save frames as .png images in the `data/IDDPedestrian/images` directory

## Intention Prediction

### Setup
1. Create the conda environment using the provided configuration file:
```bash
conda env create -f envs/intention_config.yml
conda activate baseline
```

### Training and Testing
Run the following script to train and test the intention prediction model:
```bash
cd Intention
bash run_all_on_iddped.sh
```

## Trajectory Prediction

### PIEPredict
1. Setup environment:
```bash
conda env create -f envs/PIEPredict_config.yml
conda activate traj
```

2. Run prediction:
```bash
cd PIEPredict
python train_test.py 2
```

### MTN (Multiple Trajectory Network)
1. Setup environment:
```bash
conda env create -f envs/MTN_config.yml
conda activate MTN
```

2. Run prediction:
```bash
cd MTN
python train.py
```

### BiTraP
1. Setup environment:
```bash
conda env create -f envs/bitrap_config.yml
conda activate bitrap
```

2. Testing:
```bash
cd BITRAP
python tools/test.py --config_file configs/bitrap_np_IDDP.yml CKPT_DIR epoch_latest.pth
```

3. Training and testing:
```bash
cd BITRAP
python tools/train.py --config_file configs/bitrap_np_IDDP.yml CKPT_DIR epoch_latest.pth
```

### SGNet
1. Setup environment:
```bash
conda env create -f envs/SGNet_config.yml
conda activate SGNet
```

2. Run prediction:
```bash
cd SGNet
```

## Citation
If you find this work useful, please cite our paper:
```bibtex
[Citation placeholder]
```
