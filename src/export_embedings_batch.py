# -*- coding: utf-8 -*-

"""
use pre-trained `Inception Resnet v1` as a image feature extractor.
refer to: https://github.com/davidsandberg/facenet
"""
import sys
import os
import pdb
import re
import argparse
import time

import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

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

def main(args):
    # obtain embedding
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # load model
            model_path = os.path.join(os.getcwd(),"model/20180402-114759")
            meta_file,ckpt_file = get_model_filenames(model_path)
            saver = tf.train.import_meta_graph(os.path.join(model_path,meta_file))
            saver.restore(tf.get_default_session(),os.path.join(model_path,ckpt_file))
            image_size = 160
            batch_size = 1
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            print("embedding loaded:",embeddings)
            
            # find how many images are under the input dir
            class_files = os.listdir(args.input_dir)
            nrof_images = 0
            for cf in class_files:
                nrof_images += len(os.listdir(os.path.join(args.input_dir,cf)))

            # image generator
            imagen = ImageDataGenerator(rescale=1./255)
            imgs = imagen.flow_from_directory(args.input_dir,
                target_size=(image_size,image_size),
                batch_size = batch_size,
                shuffle=False,
                color_mode = "rgb",
                class_mode = "categorical",
                classes=class_files,
                )

            # load images
            num_processed = 0
            for i,(img,y) in enumerate(imgs):
                # pdb.set_trace()
                feed_dict = {images_placeholder:img,phase_train_placeholder:False}
                res= sess.run(embeddings,feed_dict=feed_dict)
                if i == 0:
                    emb_ar = res
                else:
                    emb_ar = np.r_[emb_ar,res]
                num_processed += batch_size
                print("processed imgs:",num_processed,
                        "get embeddings with shape",res.shape)
                if num_processed >= nrof_images:
                    print("processed Done!")
                    break


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.save(os.path.join(args.output_dir,
        "embdata_%s.npy"%(time.strftime("%Y%m%d_%H_%M_%S"))),emb_ar)

    print("export complete!")
        
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",type=str,
            help="input images directory.")
    parser.add_argument("--output_dir",type=str,
            default="faces/people/emb_data",
            help="output image embeddings dir")

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))