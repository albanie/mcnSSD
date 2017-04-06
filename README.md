## Single Shot MultiBox Detector

This directory contains code to train and evaluate the SSD object detector
described in the paper:

```
SSD: Single Shot MultiBox Detector
by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, 
Scott Reed, Cheng-Yang Fu, Alexander C. Berg
```

The code is based on the `caffe` implementation 
(made available)[https://github.com/weiliu89/caffe/tree/ssd] by Wei Liu.

### Performance

The `matconvnet` training code aims to reproduce the results
achieved by the `caffe` training routine.  Using the "zoom out"
data augmentation scheme described in the updated SSD paper
the model trained with `matconvnet` achieves
a similar mAP on the 2007 test set to the `caffe` model.

```
Test Set Results:  

Comparison of the ssd-pascal-vggvd-300 model

-------------- -------------------- ------------------------
               trained with caffe   trained with matconvnet
-------------- -------------------- ------------------------
aeroplane      80.53                82.39
bicycle        83.77                85.82
bird           76.40                77.40
boat           71.53                71.43
bottle         50.17                52.82
bus            86.90                86.54
car            86.05                86.20
cat            88.57                87.04
chair          59.96                60.07
cow            81.39                81.59
diningtable    76.30                75.57
dog            85.92                84.65
horse          86.60                86.65
mean           77.54                77.75
motorbike      83.62                84.94
person         79.57                79.47
pottedplant    52.62                50.30
sheep          79.22                79.19
sofa           78.89                78.82
train          86.52                87.02
tvmonitor      76.31                77.15
-------------- -------------------- ------------------------
mean           77.54                77.75
-------------- -------------------- ------------------------
```

### Speed

The pretrained `ssd-pascal-vggvd-300` model runs at approximately 50 Hz on a titan X.
