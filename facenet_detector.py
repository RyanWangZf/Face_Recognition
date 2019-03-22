# -*- coding: utf-8 -*-
# Written by Zifeng
# wangzf18@mails.tsinghua.edu.cn

"face embeddings extraction for verification."
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import time

import tensorflow as tf 
import numpy as np

# tf.app.DEFINE_string("input_dir","./data",
#     "Used stored embeddings data in *.npy.")

# tf.app.flags.DEFINE_float("gpu_memory_fraction",0.1,
#     "Upper bound on the amount of GPU memory that will be used by the process.")
# tf.app.flags.DEFINE_string("model_path","./ckpt/facenet/20180402-114759/model-20180402-114759.ckpt-275",
#     "Saved Inception-Resnet-v1 model checkpoint.")
# FLAGS = tf.app.flags.FLAGS

ckpt_path = "./ckpt/facenet/20180402-114759/model-20180402-114759.ckpt-275"
meta_path = "./ckpt/facenet/20180402-114759/model-20180402-114759.meta" 
emb_path = "./data/face_emb.npy"
img_size = 160

def face_verify(face):
    print("Input face shape:",face.shape)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess,ckpt_path)

            # get placeholds
            img_plhd = tf.get_default_graph().get_tensor_by_name("input:0")
            emb_plhd = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            is_train_plhd = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # get embeddings
            feed_dict = {img_plhd:face, is_train_plhd:False}
            res = sess.run(emb_plhd,feed_dict=feed_dict)







if __name__ == '__main__':








