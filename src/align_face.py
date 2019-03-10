# -*- coding: utf-8 -*-
import pdb
import cv2 
import dlib 
import sys 
import numpy as np 
import os
import argparse

def main(args=None):

    # 获取当前路径 
    current_path = os.getcwd() 
    # 指定你存放的模型的路径
    # predicter_path = current_path + '/model/shape_predictor_68_face_landmarks.dat' 
    predicter_path = current_path + '/model/shape_predictor_5_face_landmarks.dat'
    # 检测人脸特征点的模型放在当前文件夹中
    face_file_path = os.path.join(current_path,args.image_path)

    # 要使用的图片，图片放在当前文件夹中 
    print(predicter_path)
    print(face_file_path)
     # 导入人脸检测模型 
    detector = dlib.get_frontal_face_detector() 
    # 导入检测人脸特征点的模型 
    sp = dlib.shape_predictor(predicter_path) 
    # 读入图片 
    bgr_img = cv2.imread(face_file_path) 
    if bgr_img is None: 
        print("Sorry, we could not load '{}' as an image".format(face_file_path)) 
        exit() 
    # opencv的颜色空间是BGR，需要转为RGB才能用在dlib中 
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) 
    # 检测图片中的人脸 
    dets = detector(rgb_img, 1) 
    # 检测到的人脸数量 
    num_faces = len(dets) 
    if num_faces == 0: 
        print("Sorry, there were no faces found in '{}'".format(face_file_path)) 
        exit() 
    # 识别人脸特征点，并保存下来
    faces = dlib.full_object_detections() 
    for det in dets: 
        faces.append(sp(rgb_img, det)) 
    # 人脸对齐 
    images = dlib.get_face_chips(rgb_img, faces, size=320) 
    # 显示计数，按照这个计数创建窗口 
    image_cnt = 0

    # 显示对齐结果 
    for image in images: 
        image_cnt += 1 
        cv_rgb_image = np.array(image).astype(np.uint8) # 先转换为numpy数组 
        cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR) # opencv下颜色空间为bgr，所以从rgb转换为bgr 
        save_path = os.path.join(args.save_dir,"%s.png"%str(image_cnt))
        # pdb.set_trace()
        cv2.imwrite(save_path,cv_bgr_image)

        # pdb.set_trace()
        # cv2.imshow('%s'%(image_cnt), cv_bgr_image) 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        exit()

    cv2.destroyAllWindows()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path",type=str,
            help="input images path.")
    parser.add_argument("--save_dir",type=str,
        default="faces/raw_face",
            help="output images path.")

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))