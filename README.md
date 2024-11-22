# IDDPeD: Pedestrian Intention and Trajectory Prediction in Unstructured Traffic Environments

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
from iddped_interface import iddped
dataset = iddped()
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
bash run_all_on_iddp.sh
```

## Trajectory Prediction

### PIEPredict
1. Setup environment:
```bash
conda env create -f envs/piepred_env.yml
conda activate piepred
```

2. Run prediction:
```bash
cd PIEPredict
python train_pie_predict.py --config configs/default.yaml
```

### MTN (Multiple Trajectory Network)
1. Setup environment:
```bash
conda env create -f mtn_env.yml
conda activate mtn
```

2. Run prediction:
```bash
cd MTN
python train_mtn.py --dataset iddped --mode train
```

### BiTraP
1. Setup environment:
```bash
conda env create -f bitrap_env.yml
conda activate bitrap
```

2. Run prediction:
```bash
cd BiTraP
python main.py --config configs/iddped_config.yaml
```

### SGNet
1. Setup environment:
```bash
conda env create -f sgnet_env.yml
conda activate sgnet
```

2. Run prediction:
```bash
cd SGNet
python train_sgnet.py --data_path data/iddped --batch_size 32
```

## Citation
If you find this work useful, please cite our paper:
```bibtex
[Citation placeholder]
```
