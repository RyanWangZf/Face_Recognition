# -*- coding: utf-8 -*-
"real-time face detection, sex classification and face verification"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import re
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
import time
import cv2
import dlib
from sklearn.metrics.pairwise import cosine_similarity
import pdb

class facesex_cls(object):
    def __init__(self,save_model_dir):
        self.graph = tf.Graph() # construct a graph for this network
        self.sess = tf.Session(graph=self.graph) # create new session
        with self.graph.as_default():
            # load model
            self.x,self.y,self.phase_train = self.load_sign_model(self.sess,save_model_dir)

    def predict(self,face):
        face = np.expand_dims(face,axis=0) / 255.
        # pdb.set_trace()
        label = self.sess.run(self.y,feed_dict={self.x:face,self.phase_train:False})
        return np.argmax(label,1)[0]

    def load_sign_model(self,sess,save_model_dir):
        input_key = "input_x"
        phase_train_key = "phase_train"
        output_key="output"
        signature_key = "signature"
        meta_graph_def = tf.saved_model.loader.load(sess,
            tags=["sex_cls_model"],
            export_dir=save_model_dir,
            )
        # obtain SignatureDef object from meta_graph_def
        signature = meta_graph_def.signature_def

        phase_train_name = signature[signature_key].inputs[phase_train_key].name
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name

        # obtain tensor and inference
        phase_train = sess.graph.get_tensor_by_name(phase_train_name)
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)

        return x,y,phase_train

class person_cls(object):
    def __init__(self,save_model_dir,save_people_dir):
        self.graph = tf.Graph() # construct a graph for this network
        self.sess = tf.Session(graph=self.graph) # create new session
        with self.graph.as_default():
            self.x,self.phase_train,self.embeddings = self.load_model(self.sess,save_model_dir)
        self.save_people_dir = save_people_dir

    def predict(self,face):
        face = np.expand_dims(face,axis=0)/255.
        emb = self.sess.run(self.embeddings,
        		feed_dict={self.x:face,self.phase_train:False})
        # pdb.set_trace()
        person_data_path =  os.path.join(self.save_people_dir,"emb_data")
        person_data = np.load(os.path.join(person_data_path,os.listdir(person_data_path)[0]))
        person_name = os.listdir(os.path.join(self.save_people_dir,"images"))
        sim_mat = cosine_similarity(np.r_[emb,person_data])[0,1:]

        # if similarity is too small, output unknown
        if sim_mat.max() < 0.5:
            return "Unknown"
        else:
            idx = np.argmax(sim_mat) 
            return person_name[idx]

    def load_model(self,sess,save_model_dir):
        model_path = save_model_dir
        meta_file,ckpt_file = self.get_model_filenames(model_path)
        saver = tf.train.import_meta_graph(os.path.join(model_path,meta_file))
        saver.restore(sess,os.path.join(model_path,ckpt_file))
        image_size = 160
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        # print("embedding loaded:",embeddings)
        return images_placeholder,phase_train_placeholder,embeddings

    def get_model_filenames(self,model_dir):
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files)==0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files)>1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        # pdb.set_trace()
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            return meta_file, ckpt_file
        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups())>=2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file

def main(args):
    print('Creating networks and loading parameters') 
    # if GPU memory is not enough, force it using CPU for computing
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # load person classifier
    p_cls = person_cls("model/20180402-114759",args.people_dir)

    if args.detect_sex:
        sex_cls = facesex_cls(save_model_dir='model/sex_cls')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True,allow_soft_placement=True),)
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    frame_interval = args.frame_interval # number of frames afer which to run face detection
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
                # save aligned face image
                det = dlib.rectangle(*face_box.astype(int))
                pts = pts_arr[i]
                face = sp(gray_frame,det)
                landmarks = np.matrix([[p.x,p.y] for p in face.parts()])

                # save and detect aligned images
                face = dlib.get_face_chip(rgb_img,face,size=160)

                # find person
                person_name = p_cls.predict(face)
                
                # print("[warning] cannot find person classifier.")
                # person_name = "face"

                # do face sex classification
                if args.detect_sex:
                    sex_label = sex_cls.predict(face)
                    add_overlays(frame,face_box,sex_label,person_name)
                else:
                    add_overlays(frame,face_box,1,person_name)

                # face = cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
                # cv2.imwrite("face_img_aligned/%d_img.png"%nrof_aligned_face,face)

                # put circles on points
                for idx,point in enumerate(landmarks):
                    pos = (point[0,0],point[0,1])
                    cv2.circle(frame,pos,4,color=(255,0,255))
                    cv2.putText(frame,str(idx + 1), pos,cv2.FONT_HERSHEY_SIMPLEX,0.4, 
                        (0, 255, 255),1, cv2.LINE_AA) 
                # other put Texts
                cv2.putText(frame,"num of faces:"+str(len(det_arr)),(10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=2,lineType=2)
                cv2.putText(frame,str(frame_rate)+"fps",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=2,lineType=2)
                
        frame_count += 1
        cv2.imshow("Video",frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def add_overlays(img,face_box,sex_label=-1,person_name="face"): 
    face_box = face_box.astype(int)

    if sex_label in [0,1]:
        if sex_label == 0:
            cv2.putText(img,person_name,(face_box[0],face_box[3]),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),thickness=1,lineType=2)
            cv2.rectangle(img,(face_box[0],face_box[1]),(face_box[2],face_box[3]),(0,97,255),2)
        elif sex_label == 1:
	        cv2.putText(img,person_name,(face_box[0],face_box[3]),
	                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),thickness=1,lineType=2)
	        cv2.rectangle(img,(face_box[0],face_box[1]),(face_box[2],face_box[3]),(255,0,195),2)

    else: # no sex classifier loaded
        cv2.putText(img,person_name,(face_box[0],face_box[3]),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),thickness=1,lineType=2)
        cv2.rectangle(img,(face_box[0],face_box[1]),(face_box[2],face_box[3]),(0,255,0),2)

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

    parser.add_argument("--people_dir",type=str,
    	help="Directory of people faces.",default="faces/people")

    parser.add_argument("--gpu_memory_fraction",type=float,
        help="Upper bound on the amount of GPU memory that will be used by the process.",
        default = 1.0)
    parser.add_argument("--detect_multiple_faces",type=bool,
        help="Detect and align multiple faces per image.",default=True)

    parser.add_argument("--detect_sex",type=bool,
    	help="Detect face and its sex.",default=False)

    parser.add_argument("--frame_interval",type=int,
        help="Detect face per number of frames.",default=3)
    
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
