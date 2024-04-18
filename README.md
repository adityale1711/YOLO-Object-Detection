# YOLO Object Detection for Ball Possession Tracking

This project implements Object Detection using YOLO with SORT Algorithm to track ball position.
Here, I've provide steps to train and run our model to implement it in the context of ball possession tracking in Football game.

## About the Project

This project aims to provide an efficient solution for counting ball possession in sports such as football or ther games.
I employ an Object Detection approach using YOLO to detect ball, inter milan and ac milan player in this case
and then apply the SORT algorithm to track ball movements.

## Installation

1. System Requirements:
   - Python 3.8
   - CUDA if using GPU for faster training and inference
   - yaml
   - ultralytics
   - roboflow
   - opencv-python
   - numpy
   - scikit-learn
2. Clone this repository:

```bash
git clone https://github.com/adityale1711/YOLO-Object-Detection-for-Ball-Possession-Counter.git
```

3. Clone the SORT Algorithm repository from [nwojke/deep_sort](https://github.com/nwojke/deep_sort):
```bash
git clone https://github.com/nwojke/deep_sort.git
```

## Usage
1. Training the model:
   - Uncomment downloadDatasets() if you didn't have the datasets
   - Run `train.py`
   - Load pretrained model for transfer weight,you can download pretrained model [here](https://docs.ultralytics.com/models/yolov8/#performance-metrics).
   - Load `data.yaml` file from datasets
   - Load `train`, `val` and `test` folder from datasets
   - Wait until train completed

2. Inference model:
   - Copy model path to model variable
   - Copy video path to `cap` variable
   - Write result filename to `output_vid` variable
   - run `detect.py`

## Result

https://github.com/adityale1711/YOLO-Object-Detection-for-Ball-Possession-Counter/assets/72447020/5db9223d-0733-4f17-ad18-79bb167d503c

