# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import pdb
import re
import glob
import cv2
import numpy as np
# from tensorflow.python import pywrap_tensorflow

# force using CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def get_model_filenames(model_dir):
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

# obtain embedding
with tf.Graph().as_default():
    with tf.Session() as sess:

        # load model
        model_path = os.path.join(os.getcwd(),"model/20180402-114759")
        meta_file,ckpt_file = get_model_filenames(model_path)
        saver = tf.train.import_meta_graph(os.path.join(model_path,meta_file))
        saver.restore(tf.get_default_session(),os.path.join(model_path,ckpt_file))
        
        image_size = 160
        
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        print("embedding loaded:",embeddings)
        
        # load images
        images = glob.glob("faces/*.png")
        img_ar = []
        for img in images:
            img_ = cv2.imread(img,cv2.IMREAD_COLOR)
            img_ = cv2.resize(img_,(image_size,image_size))
            img_ = cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)
            img_ar.append(img_)
            
        img_ar = np.array(img_ar)
        
        # obtain embeddings
        feed_dict = {images_placeholder:img_ar,phase_train_placeholder:False}
        res = sess.run(embeddings,feed_dict=feed_dict)
        
        print("get embeddings with shape:",res.shape)
        
        # training

        """
        # print tensors value and name
        reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(model_path,ckpt_file))
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print("tensor name",key)
            # print(reader.get_tensor(key))
        """
        """
        # plot graph
        writer = tf.summary.FileWriter("summary/log.log",sess.graph)
        writer.close()
        """


