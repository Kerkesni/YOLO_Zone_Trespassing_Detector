# Zone Trespassing Detector

## Objective
Detecting when a person is inside a defined zone 

## Deep Learning Methods
### Model 1
- Pretrained [Single Shot Multibox Detector (SSD)](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#inference) model

### Dataset:
The dataset used for training is the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset.

### How it works ?
1. We define the coordonates of the zone
2. We proccess each frame in the video using the SSD model
3. We suppress all bouding boxes that are not of the class "Person"
4. We calculate the intersection between all bounding boxes and the bouding box of the zone
5. We flag bounding boxes that hare 80% inside the zone.

### Results
![results](results/output_ssd.gif)

### Model 2
- Pretrained Faster-RCNN model

### Dataset:
The dataset used for training is the [COCO](https://cocodataset.org/#home) dataset.

### How it works ?
1. We define the coordonates of the zone
2. We proccess each frame in the video
3. We suppress all bouding boxes that are not of the class "Person"
4. We calculate the intersection between all bounding boxes and the bouding box of the zone
5. We flag bounding boxes that hare 80% inside the zone.

### Results
![results](results/output_rcnn.gif)

## HOG Descriptor + SVM Method
In this alternative approach we use HOG descriptor results as features for an SVM classifier to detect humans in the video

### How it works ?
1. We define the coordonates of the zone
2. We proccess each frame
4. We calculate the intersection between all bounding boxes and the bouding box of the zone
5. We flag bounding boxes that hare 80% inside the zone.

### Resluts
![hog&svm](results/output_hog.gif)