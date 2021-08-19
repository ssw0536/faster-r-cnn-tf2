# Faster R-CNN on Tensorflow2
![Python](https://img.shields.io/badge/Python-3.7-blue)
![Ubuntu](https://img.shields.io/badge/Ubuntu-18.04-green)
![Tensorflow](https://img.shields.io/badge/Tensorflow-2.3.1-orange)
## Preparation
---
### 1. Clone the code
```bash
https://github.com/ssw0536/faster-r-cnn-tf2.git
```
### 2. Install python packages
List of required python packages.

* `opencv-python>=4.4.0.46`
* `tensorflow>=2.3.1`
* `matplotlib>=3.3.3`
* `numpy>=1.18.5`
* `seaborn>=0.11.1`

Install python packages at once.
```bash
pip install -r requirement.txt
```

### 3. Download VOC2007 dataset
```bash
source dataset/download_VOC2007.sh
```
### 4. Download pre-trained model
```bash
source model/download_model.sh
```

## Tutorials
---
1. [Dataset Explore](./01_dataset_explore.ipynb)
2. [Region Proposal Network Targets](./02_rpn_targets.ipynb)
2. [Detection Network(Fast R-CNN) Targets](./03_rcnn_targets.ipynb)
2. [Faster R-CNN Final Detections](./04_final_detection.ipynb)