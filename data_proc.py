#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:19:10 2017

@author: rsubramanian
"""
import glob
import numpy as np
from skimage import io

def make_datasets(car_parent_dir, noncar_parent_dir):
    """get paths of car and non-car images from directories.
       # Args:
           car_parent_dir(str)   : car's parent directory
           noncar_parent_dir(str): non-car's parent directory
       # Returns:
           dataset_pathes(dic): dictionary of car and non-car path
                                key: ["car", "noncar"]
    """
    datasets_dir = {}
    car_list = np.array([])
    noncar_list = np.array([])
    car_directories = glob.glob(car_parent_dir)
    for car_dir in car_directories:
        car_list = np.hstack((car_list, np.array(glob.glob(car_dir + "/*.png"))))
        if car_dir == "./vehicles/KITTI_extracted":
            car_list = np.hstack((car_list, np.array(glob.glob(car_dir + "/*.png"))))
        if car_dir == "./vehicles/GTI_Right":
            car_list = np.hstack((car_list, np.array(glob.glob(car_dir + "/*.png"))))
        if car_dir == "./vehicles/GTI_Left":
            car_list = np.hstack((car_list, np.array(glob.glob(car_dir + "/*.png"))))

    noncar_directories = glob.glob(noncar_parent_dir)
    for noncar_dir in noncar_directories:
        noncar_list = np.hstack((noncar_list, np.array(glob.glob(noncar_dir + "/*.png"))))
        if noncar_dir == "./non-vehicles/Extras":
            noncar_list = np.hstack((noncar_list, np.array(glob.glob(noncar_dir + "/*.png"))))
            noncar_list = np.hstack((noncar_list, np.array(glob.glob(noncar_dir + "/*.png"))))

    datasets_dir = {"car" : car_list, "noncar" : noncar_list}
    return datasets_dir

def input_datasets(datasets, shape=(64, 64, 3)):
    """Input images from paths of car and non-car.
       For adjust brightness, left-right balance, I apply data augmentation
       Apply Flip and Brightness conversion to GTI_Left and GTI_Right
       # Args:
           datasets(dic): paths of datasets "car" and "non-car"
           shape(tuple) : shape of input images
       # Returns:
           input_images(ndarray): all images of datasets(4 dimension)
           Y (ndarray)          : all labels of datasets(1 (car) or 0 (non-car))
    """
    left_true = glob.glob("./vehicles/GTI_Left/*.png")
    len_left_true = len(left_true)
    right_true = glob.glob("./vehicles/GTI_Right/*.png")
    len_right_true = len(right_true)
    input_images = np.zeros((datasets["car"].shape[0] + datasets["noncar"].shape[0],
        shape[0], shape[1], shape[2]), dtype=np.uint8)
    input_images[:datasets["car"].shape[0]] = [io.imread(path) for path in datasets["car"]]
    input_images[datasets["car"].shape[0]:] = [io.imread(path) for path in datasets["noncar"]]

    #augmented_images = np.zeros((len_left_true*2 + len_right_true*2,
    #    shape[0], shape[1], shape[2]), dtype=np.uint8)
    #augmented_images[:len_left_true] = [augment_brightness_image(io.imread(path)) for path in left_true]
    #augmented_images[len_left_true:len_left_true*2] = [cv2.flip(augment_brightness_image(io.imread(path), bright=1.1), 1) for path in left_true]
    #augmented_images[2*len_left_true:2*len_left_true+len_right_true] = [augment_brightness_image(io.imread(path)) for path in right_true]
    #augmented_images[2*len_left_true+len_right_true:] = [cv2.flip(augment_brightness_image(io.imread(path), bright=1.1), 1) for path in right_true]
    #input_images = np.vstack((input_images, augmented_images))
    Y = np.hstack((np.ones((datasets["car"].shape[0])), np.zeros(datasets["noncar"].shape[0])))
    Y = np.hstack((Y, np.ones((len_left_true*2 + len_right_true*2))))
    return input_images, Y