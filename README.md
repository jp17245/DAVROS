# DAVROS system - 4th Year Individual Project by James Pegg

The DAVROS system aims to detect and group individuals for use in verifying the ‘Rule of Six’ instated by the UK Government. This system makes use of the third party open source implementation of YOLOv3 by Huynh Ngoc Anh found here: https://github.com/experiencor/keras-yolo3.

The original YOLOv3: https://pjreddie.com/darknet/yolo/

<b>Convert.py</b> - is used to convert annotations from PASCAL Annotations version 1.0 to PASCAL  
VOC format. This code also resizes images by half their dimensions on each axis. This was used to reduce the training time of the model.

 <b>predict.py</b> - Modified from YOLOv3 by Huynh Ngoc Anh <br />
<b>utlils/bbox.py</b>-  Modified from YOLOv3 by Huynh Ngoc Anh <br />
<b>config.json</b> -  Modified from YOLOv3 by Huynh Ngoc Anh <br />

<b>INRIAPerson</b> folder contains the entire INRIA Person dataset. Found here: http://pascal.inrialpes.fr/data/human/. Inside the ‘Test’ and ‘Train’ folder are two additional folders created for the DAVROS system. The ‘Small’  test or train folder contains the resized versions of the original images contained in the ‘pos’ folders. The ‘SmallAnnot’ folders contain the reformatted annotations with the adjusted positions and sizes of the image.

<b>Models</b>  folder contains additional models trained for the DAVROS system.  

Current person.h5 model achives a mAP of 93% when evaluated on the INRIA Person test set.

<b>LICENSE</b> file contains the original licence from the YOLOv3 implementation by Huynh Ngoc Anh. 

In order to install the dependencies for the system:
```
pip install -r requirements.txt
```
In  order to run the code :
`python predict.py -c config.json -i /path/to/image/or/video`

The distance between individuals in the same group can be adjusted, currently it is set you use the average height of the bouding boxes scaled by the average height of a person globally divided by 200cm. However, this distance can be set manually, in terms of pixels within the image.

The number of people per group before they are flagged as breaking the rule can also be adjusted, the system is currently designed to detect groups of more than six.
