#! /usr/bin/env python

import os
import argparse
import json
import cv2
import time
from cv2.cv2 import VideoCapture

from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from utils.bbox import draw_boxesVideo
from keras.models import load_model
from tqdm import tqdm
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def _main_(args):

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output
    groups = [[[]]]
    broken = []
    newGroup= [[[]]]
    count=0


    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416# a multiple of 32, the smaller the faster 416 original - 672 for mp4
    obj_thresh, nms_thresh = 0.4,0.05 # 0.45 to start

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes 
    ###############################
    if 'webcam' in input_path: # do detection on the first webcam
        video_reader: VideoCapture = cv2.VideoCapture(0)
        frames=0
        frameCounter=0
        # the main loop
        batch_size  = 1
        images      = []
        counting=0
        while True:
            startTime= time.time()
            counting=counting+1
            ret_val, image = video_reader.read()

            if ret_val == True: images += [image]

            if (len(images)==batch_size) or (ret_val==False and len(images)>0):
                batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)
                print(len(batch_boxes))
               # _, count, groups = draw_boxesVideo(image, batch_boxes[0],['1'], obj_thresh,count,groups)
                _, count, groups = draw_boxesVideo(image, batch_boxes[0], ['1'],obj_thresh, count, groups, broken)
                #_= draw_boxes(image, batch_boxes[0], ['1'], obj_thresh)
                cv2.imshow('video with bboxes', image)
                broken = []
                newGroup = [[[]]]
                images = []
                groups = [[[]]]
                count = 0
            endTime=time.time()
            frameRate= 1 /(endTime-startTime)
            frames =frames + frameRate
            frameCounter =frameCounter+1

            if cv2.waitKey(1) == 27:
                print("FrameRate: ", frames/frameCounter)
                break  # esc to quit
        cv2.destroyAllWindows()        
    elif input_path[-4:] == '.mp4': # do detection on a video
        video_out = output_path + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'MPEG'), 50.0, (frame_w, frame_h))
        # the main loop
        batch_size  = 1
        images      = []

        start_point = 0 #%
        show_window = False
        for i in tqdm(range(nb_frames)):
        #for i in range(1):
            _, image = video_reader.read()

            if (float(i+1)/nb_frames) > start_point/100.:
                images += [image]

                if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                    for i in range(len(images)):

                        # draw bounding boxes on the image using labels
                        _,count,groups= draw_boxesVideo(images[i], batch_boxes[i], config['model']['labels'], obj_thresh,count,groups,broken)
                        #_ = draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh )
                        # show the video with detection bounding boxes          
                        if show_window: cv2.imshow('video with bboxes', images[i])

                        # write result to the output video
                        video_writer.write(images[i])
                        broken = []
                        newGroup = [[[]]]
                    images = []

                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        if show_window: cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()       
    else: # do detection on an image or a set of images
        image_paths = []
        if os.path.isdir(input_path): 
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        # for image_path in image_paths:
        #     for x in range(0,20):
        #         #obj_thresh=round(0.2*x,1)
        #         netsize=160+ (64*x)
        #         # for i in range(0, 6):
        #         #     nms_thresh=round(0.2*i,1)
        #         groups = [[[]]]
        #         count = 0
        #         image = cv2.imread(image_path)
        #         print(image_path)
        #
        #         # predict the bounding boxes
        #         boxes = get_yolo_boxes(infer_model, [image], netsize, netsize, config['model']['anchors'], obj_thresh, nms_thresh)[0]
        #
        #         # draw bounding boxes on the image using labels
        #         #_, count, groups = draw_boxes(image, boxes, config['model']['labels'], obj_thresh,count,groups)
        #         _ = draw_boxes(image, boxes, config['model']['labels'], obj_thresh)
        #         # write the image with bounding boxes to file
        #         cv2.imwrite(output_path+"Netsize:"+str(netsize) + image_path.split('/')[-1], np.uint8(image))

        for image_path in image_paths:
                    groups = [[[]]]
                    count = 0
                    broken=[]
                    image = cv2.imread(image_path)
                    print(image_path)

                    # predict the bounding boxes
                    boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

                    # draw bounding boxes on the image using labels
                    #  _, count, groups = draw_boxes(image, boxes, config['model']['labels'], obj_thresh,count,groups)
                   # _, count, groups = draw_boxes(image, boxes, config['model']['labels'], obj_thresh,count,groups,broken)
                    _, count, groups, newGroup = draw_boxesVideo(image, boxes, ['1'], obj_thresh, count, groups, broken, newGroup)
                    #_ = draw_boxes(image, boxes, config['model']['labels'], obj_thresh)
                    # write the image with bounding boxes to file
                    cv2.imwrite(output_path + image_path.split('/')[-1],np.uint8(image))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)
