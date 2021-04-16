import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.model import MobileNet
from src.data import get_data
from src.data import get_image_value

dim = (224,224)
model_type = 'Mobilenet'
bw = False

train_paths, train_labels = get_data('train')
train_images = np.array([get_image_value(i, dim, bw, model_type) for i in train_paths])
train_dict = dict(images = train_images, labels = train_labels)


valid_paths, valid_labels = get_data('valid')
valid_images = np.array([get_image_value(i, dim, bw, model_type) for i in valid_paths])
valid_dict = dict(images = valid_images, labels = valid_labels)

test_paths, test_labels = get_data('test')
test_images = np.array([get_image_value(i, dim, bw, model_type) for i in test_paths])
test_dict = dict(images = test_images, labels = test_labels)

data_dir = './data/images/syn/'
img_height = 224
img_width = 224
batch_size = 64

x_train = train_images
y_train = train_labels
x_valid = valid_images
y_valid = valid_labels
x_test = test_images
y_test = test_labels

model = MobileNet(input_shape = (224,224,3))

augmentation =ImageDataGenerator(rotation_range = 20, width_shift_range = .2, height_shift_range = .2, 
                                                       horizontal_flip = True, shear_range = .15, 
                                 fill_mode = 'nearest', zoom_range = .15)

augmentation.fit(x_train)

mobilenet_history = model.model.fit_generator(augmentation.flow(x_train, y_train, batch_size = model.batch_size),
                                              epochs = 500, #model.epochs, 
                                              callbacks = [#model.early_stopping, 
                                                           model.model_checkpoint, 
                                                           model.lr_plat,
                                                           model.tensorboard_history],
                                              validation_data = (x_valid, y_valid),
                                              verbose= 1)
                                              