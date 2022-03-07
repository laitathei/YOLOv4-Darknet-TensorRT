### YOLOv4-Darknet-TensorRT
### 1 Training in Darknet
### 1.1 Download YOLOv4-Darknet
```
git clone https://github.com/AlexeyAB/darknet
```

### 1.2 Change YOLOv4-Darknet configuration
```
cd darknet/
gedit Makefile


GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=0
OPENMP=0
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0

# set GPU=1 and CUDNN=1 to speedup on GPU
# set CUDNN_HALF=1 to further speedup 3 x times (Mixed-precision on Tensor Cores) GPU: Volta, Xavier, Turing and higher
# set AVX=1 and OPENMP=1 to speedup on CPU (if error occurs then set AVX=0)
# set ZED_CAMERA=1 to enable ZED SDK 3.0 and above
# set ZED_CAMERA_v2_8=1 to enable ZED SDK 2.X

make
```

### 1.3 Build the data folder structure
```
cd /home/laitathei/Desktop/darknet
mkdir VOCdevkit
cd VOCdevkit
mkdir VOC2007
cd VOC2007
mkdir Annotations
mkdir ImageSets
mkdir JPEGImages
cd ImageSets
mkdir Main
```

### 1.4 Seperate dataset to train,test,val and Change the VOC format to YOLO format
```
mv /home/laitathei/Desktop/darknet/voc2yolo4.py /home/laitathei/Desktop/darknet/VOCdevkit/VOC2007
cd /home/laitathei/Desktop/darknet/VOCdevkit/VOC2007
python3 voc2yolo4.py
mv /home/laitathei/Desktop/darknet/voc_annotation.py /home/laitathei/Desktop/darknet/VOCdevkit
cd /home/laitathei/Desktop/darknet/VOCdevkit
python3 voc_annotation.py
```

### 1.5 Change ```voc.names``` setting
```
cp /home/laitathei/Desktop/darknet/data/voc.names /home/laitathei/Desktop/darknet/VOCdevkit
gedit voc.names

obstacle
human
injury
```

### 1.5 Change ```voc.data``` setting
```
cp /home/laitathei/Desktop/darknet/cfg/voc.data /home/laitathei/Desktop/darknet/VOCdevkit
gedit voc.data

classes= 3
train  = /home/laitathei/Desktop/darknet/VOCdevkit/2007_train.txt
valid  = /home/laitathei/Desktop/darknet/VOCdevkit/2007_test.txt
names = /home/laitathei/Desktop/darknet/VOCdevkit/voc.names
backup = /home/laitathei/Desktop/darknet/backup/
```

### 1.6 Change ```yolov4-tiny.cfg``` setting, Remember change [convolutional] & [yolo] in line 226 and 270
```
cp /home/laitathei/Desktop/darknet/cfg/yolov4-tiny.cfg /home/laitathei/Desktop/darknet/VOCdevkit
gedit yolov4-tiny.cfg

[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch=64
subdivisions=16
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000

max_batches = 6000    # classes*2000
policy=steps
steps=4800,5400       # 80% and 90% of max_batches
scales=.1,.1

[convolutional]
size=1
stride=1
pad=1
filters=255          # 3*(classes +5)
activation=linear



[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=3            # your dataset classes
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
resize=1.5
nms_kind=greedynms
beta_nms=0.6
#new_coords=1
#scale_x_y = 2.0
```

### 1.7 Download YOLO weight
```
# YOLOv4-tiny
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
# YOLOv4
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

### 1.8 Data folder structure
```
├── VOCdevkit
   ├── VOC2007
       ├── Annotations
       │   ├── xxx.xml
       │   ├──   ...
       │   
       ├── ImageSets
       │   ├── Main
       │       ├── test.txt
       │       ├── train.txt
       │       ├── trainval.txt
       │       ├── val.txt
       │   
       ├── labels
       │   ├── xxx.txt
       |   ├──   ...
       │   
       ├── JPEGImages
       │   ├── xxx.jpg
       |   ├──   ...
       │   
       └── voc2yolo4.py
   ├── 2007_train.txt
   ├── 2007_test.txt
   ├── 2007_valid.txt
   ├── train.all.txt
   ├── train.txt
   ├── voc.data
   ├── voc.names
   ├── voc_annotation.py
   └── yolov4-tiny.cfg
```

### 1.9 Training
```
cd..
./darknet partial cfg/yolov4-tiny.cfg yolov4-tiny.weights yolov4-tiny.conv.29 29    # optional
./darknet detector train VOCdevkit/voc.data VOCdevkit/yolov4-tiny.cfg yolov4-tiny.conv.29

## Below content will show if program success
 Tensor Cores are used.

 6000: 0.062273, 0.062858 avg loss, 0.000026 rate, 0.380254 seconds, 384000 images, 0.010664 hours left
Saving weights to /home/laitathei/Desktop/darknet/backup//yolov4-tiny_6000.weights
Saving weights to /home/laitathei/Desktop/darknet/backup//yolov4-tiny_last.weights
Saving weights to /home/laitathei/Desktop/darknet/backup//yolov4-tiny_final.weights
If you want to train from the beginning, then use flag in the end of training command: -clear
```

### 1.10 Evaluates Trained weight performance
```
./darknet detector map VOCdevkit/voc.data VOCdevkit/yolov4-tiny.cfg backup/yolov4-tiny_last.weights

## Below content will show if program success
class_id = 0, name = obstacle, ap = 0.00%   	 (TP = 0, FP = 0) 
class_id = 1, name = human, ap = 34.97%   	 (TP = 239, FP = 2) 
class_id = 2, name = injury, ap = 34.86%   	 (TP = 41, FP = 0) 

 for conf_thresh = 0.25, precision = 0.99, recall = 0.29, F1-score = 0.44 
 for conf_thresh = 0.25, TP = 280, FP = 2, FN = 698, average IoU = 83.42 % 

 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
 mean average precision (mAP@0.50) = 0.232763, or 23.28 % 
Total Detection Time: 3 Seconds
```

### 1.11 Inference with C++
```
# YOLOv4-tiny Video
./darknet detector demo VOCdevkit/voc.data VOCdevkit/yolov4-tiny.cfg backup/yolov4-tiny_last.weights /home/laitathei/Desktop/video_camera_color_image_raw.mp4 -out_filename /home/laitathei/Desktop/results1.mp4

# YOLOv4-tiny image
./darknet detector test ./cfg/coco.data cfg/yolov4-tiny.cfg yolov4-tiny.weights data/dog.jpg
```

### 1.12 Inference with Python
```
# YOLOv4-tiny Video
python3 darknet_video.py --input /home/laitathei/Desktop/video_camera_color_image_raw.mp4 --out_filename /home/laitathei/Desktop/results1.mp4 --weights backup/yolov4-tiny_last.weights --config_file VOCdevkit/yolov4-tiny.cfg --data_file VOCdevkit/voc.data

# YOLOv4-tiny Image
python3 darknet_images.py --input /home/laitathei/Desktop/darknet/data/dog.jpg --weights yolov4-tiny.weights --config_file VOCdevkit/yolov4-tiny.cfg --data_file cfg/coco.data
```

### 1.13 Inference with ROS, Realsense, Python
```
python3 inference_ros.py --weights backup/yolov4-tiny_last.weights --config_file VOCdevkit/yolov4-tiny.cfg --data_file VOCdevkit/voc.data
```

### 2 TensorRT conversion
### 2.1 Download dependency
```
sudo apt-get update
sudo apt-get install -y build-essential libatlas-base-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python3-pip
pip3 install numpy
pip3 install Cython
pip3 install pycuda --user
pip3 install onnx==1.4.1

git clone https://github.com/jkjung-avt/tensorrt_demos
```

### 2.2 Convert Darknet model to ONNX
```
cd /home/laitathei/Desktop/tensorrt_demos/plugins
make
cd /home/laitathei/Desktop/tensorrt_demos/yolo
cp /home/laitathei/Desktop/darknet/backup/yolov4-tiny_last.weights /home/laitathei/Desktop/tensorrt_demos/yolo/yolov4-tiny_last.weights
cp /home/laitathei/Desktop//darknet/VOCdevkit/yolov4-tiny.cfg /home/laitathei/Desktop/tensorrt_demos/yolo/yolov4-tiny_last.cfg
python3 yolo_to_onnx.py -m yolov4-tiny_last

## Below content will show if program success
Checking ONNX model...
Saving ONNX file...
Done.
```

### 2.3 Convert ONNX to TensorRT
```
python3 onnx_to_tensorrt.py -m yolov4-tiny_last

## Below content will show if program success
Completed creating engine.
Serialized the TensorRT engine to file: yolov4-tiny_last.trt
```

### 2.4 Inference with Python
```
# YOLOv4-tiny webcam
python3 trt_yolo.py --usb 0 -m yolov4-tiny_last
```

### 2.5 Inference with ROS, Realsense, Python
```
# YOLOv4-tiny
python3 inference_ros_trt.py -m yolov4-tiny_last -c 3
```
