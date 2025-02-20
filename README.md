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

# Annotations in IDD-PeD Dataset

The IDD-PeD dataset contains detailed annotations capturing spatial, behavioral, scene, interaction, and location information, specifically designed for understanding pedestrian behavior in unstructured traffic environments. The annotations are organized into the following categories:

## 1. Spatial Annotations
Bounding boxes and occlusion levels are annotated for pedestrians and relevant traffic objects using CVAT. Each bounding box is associated with an occlusion label:
- **Occlusion Levels:**
  - `0` - None (fully visible, <25% occluded)
  - `1` - Partially occluded (25%-75% occluded)
  - `2` - Fully occluded (>75% occluded)

### Annotated Object Types:
- **Pedestrian**
- **Vehicle** – {`car`, `motorcycle`, `bicycle`, `auto`, `bus`, `cart`, `truck`, `other`}
- **Traffic Light** – {`pedestrian`, `vehicle`}
  - **State:** {`red`, `orange`, `green`}
- **Bus Station**
- **Crosswalk**

Additionally, **2D Pose Annotations** for pedestrians are extracted using MMPose, consisting of 17 body keypoints.

---

## 2. Behavioral Annotations
Frame-level attributes capturing pedestrian behavior and intentions in unstructured environments:

### (i) **Crossing Behavior**
- `CU` – Crossing Undesignated (not following rules)
- `CFU` – Crossing Fast Undesignated
- `CD` – Crossing Designated
- `CFD` – Crossing Fast Designated
- `CI` – Crossing Irrelevant (not in the path of the ego-vehicle)
- `N/A` – Not applicable

### (ii) **Traffic Interaction**
- `WTT` – Weaving Through Traffic
- `HG` – Hand Gesture
- `Other` – Other forms of interaction
- `N/A` – Not applicable

### (iii) **Pedestrian Activity**
- `Walking` – Walking along the road
- `MS` – Moving Slowly (e.g., strolling)
- `N/A` – Not applicable

### (iv) **Attention Indicators**
- `LOS` – Looking Over Shoulder (aware of ego-vehicle)
- `FTT` – Facing Towards Traffic
- `NL` – Not Looking (distracted)
- `DB` – Distracted by Phone/Bag/Companion

### (v) **Social Dynamics**
- `GS` – Group Standing
- `CFA` – Child with Adult
- `AWC` – Adult with Child
- `N/A` – Not applicable

### (vi) **Stationary Behavior**
- `Sitting`
- `Standing`
- `IWA` – Interacting with Agents (e.g., street vendors)
- `Other`
- `N/A`

---

## 3. Scene Annotations
Environmental context attributes describing the surroundings:
- **Intersection Type:** {`NI` (No Intersection), `U-turn`, `T-right`, `T-left`, `four-way`, `Y-intersection`}
- **Signalized Type:** {`N/A`, `C` (Crosswalk), `S` (Traffic Signal), `CS` (Crosswalk + Signal)}
- **Road Type:** {`main`, `secondary`, `street`, `lane`}
- **Location Type:** {`urban`, `rural`, `commercial`, `residential`}
- **Motion Direction:** {`OW` (One-way), `TW` (Two-way)}
- **Time of Day:** {`day`, `night`}

---

## 4. Interaction Annotations
Capturing pedestrian-vehicle interactions in unstructured environments:
- **Interaction Flag:** `0` – No interaction, `1` – Interaction with ego-vehicle

---

## 5. Location Annotations
Providing spatial context to situate pedestrians within the road environment:
- `Near Divider`
- `Side of the Road`
- `Near Crosswalk`
- ...

The complete list is as follows:
![image](https://github.com/user-attachments/assets/c1c8df40-8f1f-4a04-b391-9fa2027bed23)


---

## 6. Additional Pedestrian Attributes
Further details for pedestrian tracks:
- **Crossing (in front of ego-vehicle):** {`no`: 0, `yes`: 1}
- **Age:** {`child`, `teenager`, `adult`, `senior`}
- **Gender:** {`male`, `female`, `default`}
- **Carrying Object:** {`none`, `small`, `large`}
- **Crossing Motive:** {`yes`: 1, `maybe`: 0.5, `no`: 0}
- **Crosswalk Usage:** {`yes`, `no`, `partial`, `N/A`} # Indicating the extent to which pedestrians use crosswalks when crossing.

---

## Summary of Attribute Categories

| Annotation Type      | Key Attributes                                                                                                                   |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Spatial              | Bounding Boxes, Occlusion, Object Types (Pedestrian, Vehicle, Traffic Light, Crosswalk, Bus Station), 2D Pose                    |
| Behavioral           | Crossing Behavior, Traffic Interaction, Pedestrian Activity, Attention Indicators, Social Dynamics, Stationary Behavior          |
| Scene                | Intersection Type, Signalized Type, Road Type, Location Type, Motion Direction, Time of Day                                      |
| Interaction          | Interaction Flag                                                                                        |
| Location             | Near Divider, Side of the Road, Near Crosswalk                                                                                   |
| Pedestrian Attributes| Age, Gender, Carrying Object, Crossing Motive, Crosswalk Usage, Crossing in front of ego-vehicle                                 |

---

**Note:** Examples of each annotation type are provided in the supplementary video.


## Setting up the dataset

### Cloning the repo
Clone the IDD-PeD repo and navigate to the root directory:
```bash
git clone https://github.com/Ruthvik9/DPJI.git
cd DPJI
```

### Downloading the dataset
Run the following command to download the videos of IDD-PeD:
```bash
bash download_videos.sh
```
Alternately, you can download the dataset in parts from the links provided in the download_videos.sh script, namely -
http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0001.tar
<br />
http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0002.tar
<br />
http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0003.tar
<br />
http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0004.tar
<br />
http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0005.tar
<br />
http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0006.tar
<br />
http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0007.tar
<br />
http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0008.tar
<br />
http://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0009.tar
<br />

The annotation files are included in this repo already, but can be downloaded from -
http://cvit.iiit.ac.in/images/datasets/IDDPed/Annotations/annotations.tar
<br />
http://cvit.iiit.ac.in/images/datasets/IDDPed/Annotations/annotations_vehicle.tar
<br />

Make sure the dataset is organized in the following structure - 
<br />
![image](https://github.com/user-attachments/assets/aa8e79db-fafb-4e26-bd82-561f69959269)


### Train-Test split
The videos from ['gp_set_0001','gp_set_0002','gp_set_0004','gp_set_0006','gp_set_0007'] are used to train the models and the ones
from ['gp_set_0003','gp_set_0005','gp_set_0008','gp_set_0009'] are used to test the models. We have used a roughly 70/30
train/test split, with 3284 pedestrians in the training data and 1632 pedestrians in the testing data.

Several methods require individual video frames for training and testing. Run the following code as a python script to extract frames from the videos:

```python
from iddped_interface_traj import IDDPedestrian
dataset = IDDPedestrian()
dataset.extract_and_save_images()
```
This will extract and save frames as .png images in the `data/IDDPedestrian/images` directory

### Downloading the checkpoints
Run the following command to download the checkpoints for the baseline models:
```bash
bash download_checkpoints.sh
```
Alternately, you can download the individual checkpoints from the following links -
<br />
https://mobility.iiit.ac.in/IDDPed_checkpoints/bitrap.zip
<br />
https://mobility.iiit.ac.in/IDDPed_checkpoints/intention.zip
<br />
https://mobility.iiit.ac.in/IDDPed_checkpoints/mtn.zip
<br />
https://mobility.iiit.ac.in/IDDPed_checkpoints/piefull.zip
<br />
https://mobility.iiit.ac.in/IDDPed_checkpoints/sgnet.zip
<br />

## Intention Prediction

#### Setup
1. Create the conda environment using the provided configuration file and navigate to the directory:
```bash
conda env create -f envs/intention_config.yml
conda activate baseline
cd Intention
```

#### Testing
2. For testing using pre-trained weights, use:
```bash
python test_model.py <saved_files_path>
```
For example,
```bash
python test_model.py models/jaad/MASK_PCPA/xxxx/
```


#### Training and Testing
3. To train and test all the models:
```bash
bash run_all_on_iddped.sh
```

## Trajectory Prediction

### PIEPredict
#### Setup
1. Create the conda environment using the provided configuration file and navigate to the directory:
```bash
conda env create -f envs/PIEPredict_config.yml
conda activate traj
cd PIEPredict
```

#### Testing
2. For testing using pre-trained weights, use:
```bash
python train_test.py 2
```

#### Training and testing
3. To train and test the model:
```bash
python train_test.py 1
```
This will train intention, speed and trajectory models separately and evaluate them on the test data.
Note: Training intention model uses image data and requires 32GB CPU RAM.

### MTN (Multiple Trajectory Network)
#### Setup
1. Create the conda environment using the provided configuration file and navigate to the directory:
```bash
conda env create -f envs/MTN_config.yml
conda activate MTN
cd MTN
```

#### Testing
2. For testing using pre-trained weights, use:
```bash
python test_iddped.py # Change the checkpoint path in the test_iddped.py file
```

#### Training
3. To train the model:
```bash
python train_iddped.py
```

### BiTraP
#### Setup
1. Create the conda environment using the provided configuration file and navigate to the directory:
```bash
conda env create -f envs/bitrap_config.yml
conda activate bitrap
cd BITRAP
```

#### Testing
2. For testing using pre-trained weights, use:
```bash
python tools/test.py --config_file configs/bitrap_np_IDDP.yml CKPT_DIR epoch_latest.pth
```

#### Training:
3. To train the model:
```bash
python tools/train.py --config_file configs/bitrap_np_IDDP.yml CKPT_DIR epoch_latest.pth
```

### SGNet
#### Setup
1. Create the conda environment using the provided configuration file and navigate to the directory:
```bash
conda env create -f envs/SGNet_config.yml
conda activate SGNet
cd SGNet
```

Create symlinks from the dataset path to ./data:
```bash
ln -s ./data/IDDPedestrian/ ./data/
```

#### Testing
2. For testing using pre-trained weights, use:
```bash
python tools/iddp/eval_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset IDDP --model SGNet_CVAE --checkpoint path/to/checkpoint
```
   
#### Training
3. To train the model:
```bash
python tools/iddp/train_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset IDDP --model SGNet_CVAE
```


## Citation
If you find this work useful, please cite our paper:
```bibtex
[Citation placeholder]
```

Our code is based on the implementations from the following repos:
<br />
[Pedestrian Intention Prediction](https://github.com/OSU-Haolin/Pedestrian_Crossing_Intention_Prediction)
<br />
[PIE](https://github.com/aras62/PIEPredict)
<br />
[MTN Trajectory](https://github.com/ericyinyzy/MTN_trajectory/blob/main/README.md)
<br />
[BiTraP](https://github.com/umautobots/bidirection-trajectory-predicter)
<br />
[SGNet](https://github.com/ChuhuaW/SGNet.pytorch)
<br />

Please consider citing them and supporting their work.
