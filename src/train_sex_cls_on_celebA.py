# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import pdb
import re
import glob
import cv2
import numpy as np
import argparse
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.objectives import binary_crossentropy
from sklearn import metrics

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

def build_model(emb,scope="new_layer"):
    with tf.variable_scope(scope):
        h = tf.layers.dense(inputs=emb,
            units=2,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            activation=None,
            trainable=True)
        y = tf.nn.softmax(h)

    return y

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir",type=str,
    		help="input images directory.")
    parser.add_argument("test_dir",type=str,
    		help="input images directory.")
    parser.add_argument("--emb_model_dir",type=str,
    	default="model/20180402-114759",
    	help="input pre-trained embedding model directory.")
    parser.add_argument("--n_epoch",type=int,
        default=10,
        help="num of training epochs.")
    parser.add_argument("--batch_size",type=int,
        default=64,
        help="num of batches per training.")

    return parser.parse_args(argv)

def main(args=None):
    # force using CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    input_image_size = 160
    # build graph
    tf.Graph().as_default()
    sess = tf.Session()
    # load model
    model_path = os.path.join(os.getcwd(),args.emb_model_dir)
    meta_file,ckpt_file = get_model_filenames(model_path)
    saver = tf.train.import_meta_graph(os.path.join(model_path,meta_file))
    saver.restore(sess,os.path.join(model_path,ckpt_file))

    # Get input and output tensors
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    print("embeddings built.")

    # load data
    train_datagen = ImageDataGenerator(
    	rescale=1./255,
    	shear_range=0.2,
	    rotation_range=20,
	    width_shift_range=0.2,
	    height_shift_range=0.2,
	    horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    batch_size = args.batch_size
    train_gen = train_datagen.flow_from_directory(
        os.path.join(os.getcwd(),args.train_dir),
    	target_size=(input_image_size,input_image_size),
    	batch_size=batch_size,
    	class_mode="categorical")
    test_gen = test_datagen.flow_from_directory(
        os.path.join(os.getcwd(),args.test_dir),
    	target_size=(input_image_size,input_image_size),
    	batch_size=100,
    	class_mode="categorical")

    # build network
    scope = "new_layers"
    y = build_model(embeddings,scope=scope)
    trainable_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # print(trainable_param)

    labels = tf.placeholder(tf.float32)
    loss = tf.reduce_mean(binary_crossentropy(y,labels))

    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(0.001,name="fine_tune_adam").minimize(
            loss,var_list=trainable_param)

    # pdb.set_trace()
    # only trainable params need initialized
    init_param = get_uninitialized_variables(sess)
    init = tf.variables_initializer(init_param)
    sess.run(init)

    for i in range(args.n_epoch):
        x_train,y_train = train_gen.next()
        x_test,y_test = test_gen.next()
        feed_dict = {
            images_placeholder:x_train,
            labels:y_train,
            phase_train_placeholder:True} # update batchnorm's moving averages
        _,b_loss = sess.run([train_op,loss],feed_dict=feed_dict)
        if i % 10 == 0:
            y_test_pred = sess.run(y,feed_dict={images_placeholder:x_test,phase_train_placeholder:True})
            print("epoch: {} loss: {} test_auc: {}".format(i,b_loss,metrics.roc_auc_score(y_test,y_test_pred)))

    pdb.set_trace()

def get_uninitialized_variables(sess): 
    global_vars = tf.global_variables() 
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars]) 
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f] 
    # print([str(i.name) for i in not_initialized_vars]) 
    return not_initialized_vars
 

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
















