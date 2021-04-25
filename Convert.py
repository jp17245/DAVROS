# Made by student.
import os
import cv2
import numpy as np


def findVar():
    numberPerson = 0
    persons = []
    xMins = []
    yMins = []
    xMaxs = []
    yMaxs = []

    for lines in f :


        if "Image filename" in lines:
            _,loc,_=lines.split("\"")
            _,_,fileName = loc.split("/")


        if "Image size" in lines:
            _,pos = lines.split(':')
            w,h,d = pos.split('x')
            width = int(w)
            height = int(h)
            depth = int(d)

        if "Bounding box" in lines:
            numberPerson = numberPerson +1
            _,box = lines.split(":")
            min,max = box.split("-")
            min=min.replace("(","")
            min=min.replace(")", "")
            max=max.replace("(", "")
            max=max.replace(")", "")

            xMin,yMin=min.split(",")
            xMax, yMax = max.split(",")
            xMins.append(xMin)
            yMins.append(yMin)
            xMaxs.append(xMax)
            yMaxs.append(yMax)

    return fileName,w,h,d,xMins,yMins,xMaxs,yMaxs,numberPerson

def writeXML(fileName,w,h,d,xMins,yMins,xMaxs,yMaxs,numberPerson):
    file,_ = fileName.split(".")
    print(file)
    f = open("/home/doubleitbytwo/Documents/UNI/4th-Year-Project-Jp17245/INRIAPerson/Test/SmallAnnotTest/"+file +".xml", "a")
    f.write("<annotation>")
    f.write("<folder>Annots</folder>")
    f.write("<filename>"+fileName+"</filename>")
    f.write("<path>/home/doubleitbytwo/Documents/UNI/4th-Year-Project-Jp17245/INRIAPerson/Test/SmallAnnotTest/" + fileName+"</path>")
    f.write("<source><database>Unknown</database></source>")
    f.write("<size><width>"+str(int(int(w)/2))+"</width><height>"+str(int(int(h)/2))+"</height><depth>"+d+"</depth></size>")
    f.write("<segmented>0</segmented>")
    for i in range(0,numberPerson):
        f.write("<object><name>person</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult><bndbox><xmin>"+str(int(int(xMins[i])/2))+"</xmin><ymin>"+ str(int(int(yMins[i])/2))+"</ymin><xmax>"+str(int(int(xMaxs[i])/2))+"</xmax><ymax>"+str(int(int(yMaxs[i])/2))+"</ymax></bndbox></object>")
    f.write("</annotation>")
    f.close()

def imageResize():
    #name,_ = files.split(".")
    img = cv2.imread("/home/doubleitbytwo/Documents/UNI/4th-Year-Project-Jp17245/Keras_yolo3/keras-yolo3-master/Grouping/groups/parents.png")
    #img = cv2.imread("/home/doubleitbytwo/Documents/UNI/4th-Year-Project-Jp17245/Keras_yolo3/keras-yolo3-master/ImagesOfMe/"+files)
    height =img.shape[0]
    width = img.shape[1]
    dim=(int(width/2) , int(height/2))

    resized = cv2.resize(img,dim ,interpolation = cv2.INTER_AREA)
    cv2.imwrite("/home/doubleitbytwo/Documents/UNI/4th-Year-Project-Jp17245/Keras_yolo3/keras-yolo3-master/Grouping/groups/parents.png", resized)
    #cv2.imwrite("/home/doubleitbytwo/Documents/UNI/4th-Year-Project-Jp17245/Keras_yolo3/keras-yolo3-master/ImagesOfMe/"+files,resized)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




directory = '/home/doubleitbytwo/Documents/UNI/4th-Year-Project-Jp17245/Keras_yolo3/keras-yolo3-master/ImagesOfMe/'
maxW=0;
maxH=0;
minW=10000;
minH=10000;
imageResize();
# for files in os.listdir(directory):
#     # f = open("/home/doubleitbytwo/Documents/UNI/4th-Year-Project-Jp17245/INRIAPerson/Test/annotations/"+files, "r")
#     # print(f)
#     # fileName,w,h,d,xMins,yMins,xMaxs,yMaxs,numberPerson =findVar()
#     # print("Filename: ",fileName)
#     # print(f)
#     imageResize();
#
#     writeXML(fileName,w,h,d,xMins,yMins,xMaxs,yMaxs,numberPerson)
#
#
#     # fileName,w,h,d,xMins,yMins,xMaxs,yMaxs,numberPerson =findVar()
#     # if (int(w)>maxW):
#     #     maxW=int(w)
#     # if (int(h)>maxH):
#     #     maxH=int(h)
#     # if (int(w)<minW):
#     #     minW=int(w)
#     # if (int(h)<minH):
#     #     minH=int(h)
#
#     #writeXML(fileName,w,h,d,xMins,yMins,xMaxs,yMaxs,numberPerson)
#
#
# print("Max Image Width: ",maxW)
# print("Max Image Height: ",maxH)
# print("Min Image Width: ",minW)
# print("Min Image Height: ",minH)