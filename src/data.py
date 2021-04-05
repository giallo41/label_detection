
import pickle
import cv2
import os
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import shuffle as sklearn_shuffle
from keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess

def get_image_value(path, dim, bw, model_type): 
    '''This function will read an image and convert to a specified version and resize depending on which algorithm is being used.  If edge is specified as true, it will pass the img array to get_edged which returns a filtered version of the img'''
    img = image.load_img(path, target_size = dim)
    img = image.img_to_array(img)
    if bw == True: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if model_type.upper() != 'Normal': 
            img = np.stack((img,)*3, axis =-1)
        else: 
            img = img.reshape(img.shape[0], img.shape[1],1)


    if model_type.upper() == 'MOBILENET': 
        img = mobile_preprocess(img)
        return img
    elif model_type.upper() == 'VGG16': 
        img = vgg16_preprocess(img) 
        return img
    return img/255.


def get_data(class_type): 
    true_paths = [f'./data/images/{class_type}/true/{file}' for file in os.listdir(f'./data/images/{class_type}/true')]
    true_labels = [1 for i in range(len(true_paths))]
    
    false_paths = [f'./data/images/{class_type}/false/{file}' for file in os.listdir(f'./data/images/{class_type}/false')]
    false_labels = [0 for i in range(len(false_paths))]
    
    labels = np.array(true_labels + false_labels)
    print(f'{class_type.upper()} Value Counts')
    print(pd.Series(labels).value_counts())
    paths = np.array(true_paths + false_paths)
    labels = to_categorical(labels)
    paths, labels = sklearn_shuffle(paths, labels)
    return paths, labels