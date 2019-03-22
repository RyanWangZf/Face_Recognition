# -*- coding: utf-8 -*-

"""
use pre-trained `Inception Resnet v1` as a image feature extractor.
refer to: https://github.com/davidsandberg/facenet
"""

import os
import pdb
import time

import tensorflow as tf
import numpy as np
from scipy import misc

# set params
tf.app.flags.DEFINE_string("ckpt_path","./ckpt/facenet/20180402-114759/model-20180402-114759.ckpt-275",
    "Path of pre-trained embedding extractor, checkpoint.")
tf.app.flags.DEFINE_string("meta_path","./ckpt/facenet/20180402-114759/model-20180402-114759.meta",
    "Path of pre-trained embedding extractor, meta graph.")

tf.app.flags.DEFINE_string("load_path","./data/images")
tf.app.flags.DEFINE_string("save_path","./data/face_emb.npy",
    "Path of saved face embeddings data.")

FLAGS = tf.app.flags.FLAGS

def main(_):
    files = [os.path.join(os.getcwd(),p) for p in os.listdir(FLAGS.load_path)]
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # load model
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess,ckpt_path)
            # Get input and output tensors
            images_plhd = tf.get_default_graph().get_tensor_by_name("input:0")
            emb_plhd = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            is_train_plhd = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            img_ar = []
            name_ar = []
            for i,file in enumerate(files):
                print("{}/{} img: {}".format(i,len(files),file))
                img = misc.imread(file,mode="RGB")
                img_ar.append(img)
                name = file.split("/")[-1]
                name_ar.append(os.path.splitext(name)[0])
            
            # Reshape
            img_ar = np.array(img_ar).reshape(len(img_ar),
                img_ar[0].shape[0],img_ar[0].shape[1],3)
            
            # Run inference
            feed_dict = {images_plhd:img_ar,is_train_plhd:False}
            res = sess.run(emb_plhd,feed_dict=feed_dict)






    pass

if __name__ == '__main__':
    tf.app.run()