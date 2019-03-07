from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
import time
import cv2
import dlib

import pdb

def main(args):
    print('Creating networks and loading parameters') 
    # if GPU memory is not enough, force it using CPU for computing
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True,
            allow_soft_placement=True),)
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.700 # scale factor
    frame_interval = 3 # number of frames afer which to run face detection
    fps_display_interval = 3 # seconds

    # read images
    video_capture = cv2.VideoCapture(args.input_video)
    frame_count = 0
    frame_rate = 0
    nrof_aligned_face = 0
    start_time = time.time()

    # dlib face landmark detector 
    # 5 points detection
    predictor_path = os.getcwd() + "/model/shape_predictor_5_face_landmarks.dat"
    
    # 68 points detection
    # predictor_path = os.getcwd() + "/model/shape_predictor_68_face_landmarks.dat"
    sp = dlib.shape_predictor(predictor_path)

    while True:
        # capture frame-by-frame
        ret,frame = video_capture.read()
        # scaled frame
        height,width = frame.shape[:2]
        frame = cv2.resize(frame,(int(width/2), int(height/2)),interpolation=cv2.INTER_CUBIC)

        # grey
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2BGR)

        if (frame_count % frame_interval) == 0:
            # check current fps
            end_time = time.time()
            if (end_time -start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0
            
            # detect face
            det_arr,pts_arr,flag = face_recognition(gray_frame,
                minsize,pnet,rnet,onet,threshold,factor,args)
        
        if flag: # flag == True: find faces
            nrof_aligned_face += 1
            # opencv uses BGR image while dlib uses RGB image
            rgb_img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            for i,face_box in enumerate(det_arr):
                # add overlays
                add_overlays(frame,face_box,frame_rate,nrof_faces=len(det_arr))
                
                # save aligned face image
                det = dlib.rectangle(*face_box.astype(int))
                pts = pts_arr[i]
                face = sp(gray_frame,det)
                landmarks = np.matrix([[p.x,p.y] for p in face.parts()])

                # put circles on points
                for idx,point in enumerate(landmarks):
                    pos = (point[0,0],point[0,1])
                    cv2.circle(frame,pos,4,color=(255,0,255))
                    cv2.putText(frame,str(idx + 1), pos,cv2.FONT_HERSHEY_SIMPLEX,0.4, 
                        (0, 255, 255),1, cv2.LINE_AA) 
                
                # save aligned images
                face = dlib.get_face_chip(rgb_img,face,size=160)
                face = cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
                cv2.imwrite("face_img_aligned/%d_img.png"%nrof_aligned_face,face)
               
        frame_count += 1
        cv2.imshow("Video",frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
def add_overlays(img,face_box,frame_rate,nrof_faces): 
    face_box = face_box.astype(int)
    cv2.rectangle(img,(face_box[0],face_box[1]),(face_box[2],face_box[3]),(0,255,0),2)
    
    cv2.putText(img,"face",(face_box[0],face_box[3]),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),thickness=2,lineType=2)
    
    cv2.putText(img,str(frame_rate)+"fps",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=2,lineType=2)
    
    cv2.putText(img,"num of faces:"+str(nrof_faces),(10,60),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=2,lineType=2)

def face_recognition(img,minsize,pnet,rnet,onet,threshold,factor,args):
    
    bounding_boxes, points = detect_face.detect_face(img,
        minsize,pnet,rnet,onet,threshold,factor)
    # points[:,0] are places of [left eye, right eye, nose, left mouth, right mouth]
    
    nrof_faces = bounding_boxes.shape[0]

    if nrof_faces > 0:
        det = bounding_boxes[:,0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1: # more than one face
            
            if args.detect_multiple_faces: # detect multiple faces
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            
            else: # detect only one face
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], 
                    (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-
                    offset_dist_squared*2.0) # some extra weight on the centering
                det_arr.append(det[index,:])
        
        else: # one face
            det_arr.append(np.squeeze(det))

        return det_arr,points,True
    else:
        print('Unable to find face.')
        return None,None,False
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video",type=str,help="Raw video file.")

    parser.add_argument("--gpu_memory_fraction",type=float,
        help="Upper bound on the amount of GPU memory that will be used by the process.",
        default = 1.0)
    parser.add_argument("--detect_multiple_faces",type=bool,
        help="Detect and align multiple faces per image.",default=True)

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
