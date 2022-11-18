# Source code
## Research thesis S420
The code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Requirements
The code are tested in the following environment:
- Python 3.7
- Pytorch 1.9.0
- CUDA 11.1
- spconv-cu111 v2.1.21
- waymo-open-dataset-tf-2-5-0 1.4.1
- numpy 1.19.5

## Installation
a. Clone this repository.
```shell
git clone https://github.tik.uni-stuttgart.de/iss-theses/s1420.git
```

b. Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 

c. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```

## Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](./tools/cfgs/dataset_configs),
and the model configs are located within [tools/cfgs](./tools/cfgs) for different datasets.

## Dataset Preparation

### KITTI Dataset
- The address of KITTI Dataset on the server is /data/private/m142_datasets/object .

```
/data/private/m142_datasets/object
│── ImageSets
│── training
│   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│── testing
│   ├──calib & velodyne & image_2
```

- Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### Waymo Open Dataset
* The address of Waymo dataset on the server is /data/private/m142_datasets/Waymo/waymo/waymo_format .(There are 798 *train* tfrecord and 202 *val* tfrecord ):  
```
/data/private/m142_datasets/Waymo/waymo/waymo_format
│── ImageSets
│── raw_data
│   │── segment-xxxxxxxx.tfrecord
|   |── ...
|── waymo_processed_data_v0_5_0
│   │── segment-xxxxxxxx/
|   |── ...
│── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/
│── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl
│── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional)
│── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│── waymo_processed_data_v0_5_0_infos_val.pkl (optional)
```
* Install the official `waymo-open-dataset` by running the following command: 
```shell script
pip3 install --upgrade pip
# tf 2.0.0
pip3 install waymo-open-dataset-tf-2-5-0 --user
```

* Extract point cloud data from tfrecord and generate data infos by running the following command (it takes several hours, 
and you could refer to `data/waymo/waymo_processed_data_v0_5_0` to see how many records that have been processed): 
```python 
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```

Note that you do not need to install `waymo-open-dataset` if you have already processed the data before and do not need to evaluate with official Waymo Metrics. 


## Training & Testing

### Train or test with a single GPU
- You can refer [tools/openbatch.sh](./tools/openbatch.sh)
```shell script
python train.py --cfg_file ${CONFIG_FILE} --extra_tag ${A_UNIQUE_NAME}
```

```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT} --extra_tag ${A_UNIQUE_NAME}
```

### Train or test with multiple GPUs
- You can refer [tools/multi_open.sh](./tools/multi_open.sh)

## Tips for Config files

### Change the voxel encoding

#### Voxel height

For example in the KITTI dataset:
- change the voxel height in dataset config:
```yaml
VOXEL_SIZE: [0.05, 0.05, 0.125]
```
- change the number of features in BEV map
```yaml
MAP_TO_BEV:
  NUM_BEV_FEATURES: 128
```
- The number of features is related to the total height of point cloud and the voxel height. You can refer to the combination of these 3 parameters (point cloud height, voxel height, number of features) as follow:
  - for the 3D Sparse Convolution: (4, 0.0625, 384), (4, 0.1, 256), (4, 0.125, 128)
  - for the Voxel Transformer: (4, 0.0625, 512), (4, 0.1, 320), (4, 0.125, 256), (4, 0.2, 128)

#### Voxel depth or depth

For example in the Waymo dataset:

- change the voxel depth or width in dataset config:
```yaml
VOXEL_SIZE: [0.15, 0.1, 0.15]
```
- change the point cloud range in dataset config:
```yaml
POINT_CLOUD_RANGE: [-75.6, -75.2, -2, 75.6, 75.2, 4]
```
- You can set the depth or width of the point cloud by this formula: (Point cloud depth) / (voxel depth * 16) = int

### GT-Sampling & Data filter

- If you don't want to use GT-Sampling, delete the content in the [ ] of **SAMPLE_GROUPS**:
```yaml
AUG_CONFIG_LIST:
    - NAME: gt_sampling
      USE_ROAD_PLANE: False
      DB_INFO_PATH:
          - kitti_dbinfos_train.pkl
      PREPARE: {
         filter_by_min_points: ['Car:5', 'Pedestrian:5', 'Cyclist:5'],
         filter_by_difficulty: [-1],
      }

      SAMPLE_GROUPS: []
      NUM_POINT_FEATURES: 4
      DATABASE_WITH_FAKELIDAR: False
      REMOVE_EXTRA_WIDTH: [0.0, 0.0, 0.0]
      LIMIT_WHOLE_SCENE: False
```

- If you want to filter the very sparse target, change the number in the **filter_by_min_points**:
```yaml
AUG_CONFIG_LIST:
    - NAME: gt_sampling
      USE_ROAD_PLANE: False
      DB_INFO_PATH:
          - kitti_dbinfos_train.pkl
      PREPARE: {
         filter_by_min_points: ['Car:1', 'Pedestrian:1', 'Cyclist:1'],
         filter_by_difficulty: [-1],
      }

      SAMPLE_GROUPS: ['Car:15','Pedestrian:10', 'Cyclist:10']
      NUM_POINT_FEATURES: 4
      DATABASE_WITH_FAKELIDAR: False
      REMOVE_EXTRA_WIDTH: [0.0, 0.0, 0.0]
      LIMIT_WHOLE_SCENE: False
```

### Downsampling

- In Kitti, if randomly pick 32-beams and 16-beams point cloud into training, you need to set **RANDOM_DOWNSAMPLE** to True. 
**HIGH_RESOLUTION_RATE** is the rate of 64-beams point cloud, **MID_RESOLUTION_RATE** is the rate of 32-beams point cloud.
```yaml
RANDOM_DOWNSAMPLE: True
HIGH_RESOLUTION_RATE: 0.5
MID_RESOLUTION_RATE: 0.3
```

- In Waymo, add the **random_down_sample** into AUG_CONFIG_LIST, **GENERAL_RATE** is the downsample rate for the whole point cloud,
**DENSE_RATE** is the downsample rate in the height (-0.1, 0.1).
```yaml
AUG_CONFIG_LIST:
    - NAME: random_down_sample
      GENERAL_RATE: 0.2
      DENSE_RATE: 0.5
```