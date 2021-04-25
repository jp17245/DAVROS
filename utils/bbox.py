# Original code by Huynh Ngoc Anh. Modified by student. Assign Group functions are new for the DAVROS system. DrawBoxes modified from original code.
import numpy as np
import os
import cv2
from .colors import get_color


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def assignGroupsVideo(groups, xmin, ymin, xmax, ymax, groupNum, broken,average):
    centre_x = int(xmin + ((xmax - xmin) / 2)  )# centre of new person
    centre_y = int(ymin + ((ymax - ymin) / 2))
    area_new = (xmax - xmin) * (ymax - ymin)
    newHeight= int(ymax - ymin)
    grouped = False
    change = False
    counter = -1
    hasGroup = True
    currentNewGroup=-1
    #distance= ((average /165)*200) *2
    distance = 265
   # print(distance)
    class BreakIt(Exception):
        pass
    try:
        for x in groups:
            groupPosCounter = -1
            counter = counter + 1
            if ((len(x) > 6)):  # or (counter == 0 and len(x) > 5):
                broken.append(currentNewGroup)
            for i in x:
                groupPosCounter = groupPosCounter + 1

                group_centre_x = int(i[0]) + ((int(i[2]) - int(i[0])) / 2)
                group_centre_y = int(i[1]) + ((int(i[3]) - int(i[1])) / 2)
                area_old = (i[2] - i[0]) * (i[3] - i[1])
                if ((abs(centre_x - group_centre_x) < (average/5) ) and (abs(centre_y - group_centre_y) < (average/5) )):
                    hasGroup = False
                    grouped = True
                    groupNum = counter
                    #newGroupNum =groupNum
                    posTarget = groupPosCounter
                    countingppl = 0
                    groups[counter][groupPosCounter] = [xmin, ymin, xmax, ymax]
                    if ((len(groups[counter])) > 1):
                        storeCount = 0
                        for d in groups[counter]:
                            if (storeCount != posTarget):
                                group_centre_x_check = int(d[0]) + ((int(d[2]) - int(d[0])) / 2)
                                group_centre_y_check = int(d[1]) + ((int(d[3]) - int(d[1])) / 2)
                                storeCount = storeCount + 1
                                #if (((abs(centre_x - group_centre_x) < (newHeight * (200 / 165))) and (abs(centre_y - group_centre_y) < (newHeight * (200 / 165)))) and (((area_new / area_old) < 2) and ((area_new / area_old) > 0.5))):
                                if (abs(centre_x - group_centre_x_check) < distance) and ( abs(centre_y - group_centre_y_check) < distance):
                                    countingppl = countingppl + 1
                                    #if (grouped == False):
                                     #groups[groupToAddto].append([xmin, ymin, xmax, ymax])
                                if (countingppl != (len(groups[counter])) - 1):
                                    #if (countingppl ==0):
                                    change = True
                                    hasGroup = True
                        if (countingppl == len(groups[counter]) - 1):
                            groups[counter][groupPosCounter] = [xmin, ymin, xmax, ymax]
                    else:
                        groups[counter][groupPosCounter] = [xmin, ymin, xmax, ymax]
                else:
                    if hasGroup == True and grouped == False and change == False:
                        #if (((abs(centre_x - group_centre_x) < (newHeight*(200/165))) and (abs(centre_y - group_centre_y) < (newHeight*(200/165)))) and (((area_new / area_old) < 3) and ((area_new / area_old) > 0.2))): # Values again need to scale
                        if (((abs(centre_x - group_centre_x) < distance)) and (abs(centre_y - group_centre_y) < distance)):
                            groupToAddto = counter
                           # if (grouped == False):
                              #  groups[groupToAddto].append([xmin, ymin, xmax, ymax])
                            groupNum = groupToAddto
                            #newGroupNum =groupToAddto
                            grouped = True
                            raise BreakIt
    except BreakIt:
        pass
    if (grouped == False and hasGroup == True) or (change == True and hasGroup == True):
        groups.append([])
        groups[len(groups) - 1].append([xmin, ymin, xmax, ymax])  # adding group
        groupNum = len(groups) - 1
        #newGroupNum =groupNum
    return groupNum
    # if the box boundary near each other and the  area of bounding box is near equal.


def draw_boxesVideo(image, boxes, labels, obj_thresh, count, groups,broken, quiet=True):
    average=0
    averageCounter=0
    BrokenTryAgain=[[],[]]
    BrokenTryAgainPos = [[], []]
    #print(BrokenTryAgain[0][0])
    position=0
    for box in boxes:
        if (((box.ymax - box.ymin) / (box.xmax - box.xmin)) < 5):
            label_str = ''
            label = -1
            average = average + (box.ymax -box.ymin)
            averageCounter =averageCounter +1
            #print("box")
            for i in range(len(labels)):
                if box.classes[i] > obj_thresh:
                    if label_str != '': label_str += ', '
                    label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                    label = i
                if not quiet: print(label_str)
            if label >= 0:
                #print("box")
                average = average + (box.ymax - box.ymin)
                averageCounter = averageCounter + 1

    for box in boxes:
        if(((box.ymax - box.ymin)/(box.xmax -box.xmin)) <5):   #ignores boxes with aspoect ratio grater than 5
            label_str = ''
            label = -1
            groupNum = 0
            #print("box")
            for i in range(len(labels)):
                if box.classes[i] > obj_thresh:
                    if label_str != '': label_str += ', '
                    label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                    label = i
                if not quiet: print(label_str)

            if label >= 0:
                width, height = (box.xmax - box.xmin), ((box.ymax - box.ymin))
                regionRule = np.array([[box.xmin, box.ymin],[box.xmin, box.ymin - height / 12], [box.xmin + width, box.ymin - height / 12],[box.xmin + width, box.ymin]], dtype='int32')
                count = count + 1
                if count == 1:
                    groups[0][0] = [box.xmin, box.ymin, box.xmax, box.ymax]
                    groupNum = 0
                    cv2.rectangle(img=image, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=get_color(groupNum),thickness=2)
                elif count > 1:
                    groupNum = assignGroupsVideo(groups, box.xmin, box.ymin, box.xmax, box.ymax, groupNum, broken,(average/averageCounter))
                    cv2.rectangle(img=image, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=get_color(groupNum),thickness=2)
                cv2.fillPoly(img=image, pts=[regionRule], color=get_color( groupNum))  # Used for filling rectangle above box in which label is printed on top of. To make it stand out
                cv2.putText(img=image, text="Group " + str(groupNum), org=(box.xmin + 5, box.ymin - 3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-2 * (width * 0.6), color=(0, 0, 0),thickness=1)  # Prints the label of detected object. Eg Person
                if (groupNum not in BrokenTryAgain[0]):
                    BrokenTryAgain[0].append(groupNum)
                    BrokenTryAgain[1].append(1)
                    BrokenTryAgainPos[0].append(groupNum)
                    BrokenTryAgainPos[1].append([box.xmin, box.ymin, box.xmax, box.ymax])
                    position=len(BrokenTryAgain[0])-1
                else:
                    position=BrokenTryAgain[0].index(groupNum)
                    BrokenTryAgain[1][position]= BrokenTryAgain[1][position] +1
                    BrokenTryAgainPos[0].append(groupNum)
                    BrokenTryAgainPos[1].append([box.xmin, box.ymin, box.xmax, box.ymax])

    newgroup=-5
    for x in range(len(BrokenTryAgain[0])):
        if BrokenTryAgain[1][x] > 3:
            newgroup =BrokenTryAgain[0][x]
        for i in range(len(BrokenTryAgainPos[0])):
            if BrokenTryAgainPos[0][i] == newgroup:
                xmin2=BrokenTryAgainPos[1][i][0]
                ymin2=BrokenTryAgainPos[1][i][1]
                xmax2=BrokenTryAgainPos[1][i][2]
                ymax2=BrokenTryAgainPos[1][i][3]
                posy = int(((ymax2-ymin2)/2))
                cv2.putText(img=image, text="Broke", org=(xmin2, (ymin2 + posy - 30)),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-2 * (xmax2 - xmin2), color=(0, 255, 0),thickness=1)
                cv2.putText(img=image, text="Rule", org=(xmin2, (ymin2 + posy - 10)),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-2 * (xmax2 - xmin2), color=(0, 255, 0),thickness=1)

                # cv2.putText(img=image, text="Group " + str(groupNum), org=(box.xmin + 13, box.ymin - 13),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 0, 0),thickness=2)  # Prints the label of de


    return image, count, groups



def assignGroups(groups, xmin, ymin, xmax, ymax, groupNum,average,broken):
    centre_x = xmin + ((xmax - xmin) / 2)  # centre of new person
    centre_y = ymin + ((ymax - ymin) / 2)
    grouped = False
    counter = -1
    hasGroup = True
    distance= ((average /165)*200)/2
    for x in groups:
        #groupPosCounter = -1
        counter = counter + 1
        if (len(x) >= 1): #or (counter == 0 and len(x) > 5):
            broken.append(counter)
        for i in x:
            group_centre_x = int(i[0]) + ((int(i[2]) - int(i[0])) / 2)
            group_centre_y = int(i[1]) + ((int(i[3]) - int(i[1])) / 2)
            if (abs(centre_x - group_centre_x) < distance) and (abs(centre_y - group_centre_y) < distance):
                groupToAddto = counter
                if (grouped == False):
                    groups[groupToAddto].append([xmin, ymin, xmax, ymax])
                groupNum = groupToAddto
                grouped = True
    if (grouped == False and hasGroup == True):
        groups.append([])
        groups[len(groups) - 1].append([xmin, ymin, xmax, ymax])  # adding group
        groupNum = len(groups) - 1
    return groupNum

def draw_boxes(image, boxes, labels, obj_thresh, count, groups, broken,quiet=True):
    average=0
    averageCounter=0
    for box in boxes:
        label_str = ''
        label = -1
        average = average + (box.ymax -box.ymin)
        averageCounter =averageCounter +1
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                label = i
            if not quiet: print(label_str)
        if label >= 0:
            average = average + (box.ymax - box.ymin)
            averageCounter = averageCounter + 1

    for box in boxes:
        label_str = ''
        label = -1
        groupNum = 0
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                label = i
            if not quiet: print(label_str)
        if label >= 0:
            width, height = (box.xmax - box.xmin), ((box.ymax - box.ymin))
            regionRule = np.array([[box.xmin, box.ymin],
                                   [box.xmin, box.ymin - height / 12],
                                   [box.xmin + width, box.ymin - height / 12],
                                   [box.xmin + width, box.ymin]], dtype='int32')
            count = count + 1
            if count == 1:
                groups[0][0] = [box.xmin, box.ymin, box.xmax, box.ymax]
                groupNum = 0
                cv2.rectangle(img=image, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=get_color(groupNum), thickness=2)
            elif count > 1:
                groupNum = assignGroups(groups, box.xmin, box.ymin, box.xmax, box.ymax, groupNum,(average/averageCounter),broken)
                cv2.rectangle(img=image, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=get_color(groupNum),thickness=2)
            cv2.fillPoly(img=image, pts=[regionRule], color=get_color(groupNum))  # Used for filling rectangle above box in which label is printed on top of. To make it stand out
            cv2.putText(img=image, text="Group " + str(groupNum), org=(box.xmin + 5, box.ymin - 3),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-2 * (width * 0.6), color=(0, 0, 0),thickness=1)  # Prints the label of detected object. Eg Person
            print(broken)
            if groupNum in broken:
                for i in groups[groupNum]:
                    posx = (i[2]-i[0])/2
                    posy= (i[3]-i[1])/2
                    cv2.putText(img=image, text="Broke", org=(int(i[0]), int(i[1]+posy-30)),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-2 * (i[2]-i[0]), color=(0, 255, 0),thickness=1)
                    cv2.putText(img=image, text="Rule",org=(int(i[0]), int(i[1] + posy-10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1e-2 * (i[2]-i[0]),color=(0, 255, 0), thickness=1)
            # cv2.putText(img=image, text="Group " + str(groupNum), org=(box.xmin + 13, box.ymin - 13),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 0, 0),thickness=2)  # Prints the label of de
    if(averageCounter >0):
        print(average/averageCounter)
    return image, count, groups