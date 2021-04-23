# DAVROS system - 4th Year Individual Project by James Pegg

The DAVROS system aims to detect and group individuals for use in verifying the ‘Rule of Six’ instated by the UK Government. This system makes use of the third party open source implementation of YOLOv3 by Huynh Ngoc Anh found here: https://github.com/experiencor/keras-yolo3.

The original YOLOv3: https://pjreddie.com/darknet/yolo/

Convert.py - is used to convert annotations from PASCAL Annotations version 1.0 to PASCAL  
VOC format. This code also resizes images by half their dimensions on each axis. This was used to reduce the training time of the model.

## predict.py - Modified from YOLOv3 by Huynh Ngoc Anh 
## utlils/bbox.py-  Modified from YOLOv3 by Huynh Ngoc Anh 
## config.json -  Modified from YOLOv3 by Huynh Ngoc Anh 

The INRIAPerson folder contains the entire INRIA Person dataset. Found here: http://pascal.inrialpes.fr/data/human/. Inside the ‘Test’ and ‘Train’ folder are two additional folders created for the DAVROS system. The ‘Small’  test or train folder contains the resized versions of the original images contained in the ‘pos’ folders. The ‘SmallAnnot’ folders contain the reformatted annotations with the adjusted positions and sizes of the image.

Models  folder contains additional models trained for the DAVROS system.  

LICENSE file contained the original licence from the YOLOv3 implementation by Huynh Ngoc Anh. 

In order to install the dependencies for the system:
```
pip install -r requirements.txt
```

<b>Overflow</b>

