### YOLOv4-Darknet-TensorRT
### 1 Training in Darknet
### 1.1 Download Darknet
```
git clone https://github.com/AlexeyAB/darknet
```

### 1.2 Change Darknet configuration
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
cd /home/laitathei/Desktop/darknet/VOCdevkit/VOC2007
python3 voc2yolo4.py
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

### 1.7 Data folder structure
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
       ├── voc2yolo4.py
   ├── 2007_train.txt
   ├── 2007_test.txt
   └── 2007_valid.txt
   ├── train.all.txt
   └── train.txt
   ├── voc.data
   └── voc.names
   ├── voc_annotation.py
   └── yolov4-tiny.cfg
```

### 1.8 Training
```
cd..
./darknet partial cfg/yolov4-tiny.cfg yolov4-tiny.weights yolov4-tiny.conv.29 29
./darknet detector train VOCdevkit/voc.data VOCdevkit/yolov4-tiny.cfg yolov4-tiny.conv.29
```
