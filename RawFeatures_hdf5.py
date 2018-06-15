import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Input,Dropout, Flatten, Dense
from keras import applications
from keras.applications.resnet50 import ResNet50, preprocess_input
import keras
import h5py_cache as h5c
import h5py as h5
import random
from sklearn.utils import shuffle
from math import *
import threading
print("Keras Imported")
full_data_dir = '/content/Animals/data/trainval/'
batch_size = 64
def calculateDividingFactor(shape):
    total_mem_usage = min(np.prod(shape)*4,1024**2*15000)
    total_mem_gb = total_mem_usage/(1024**3)
    dividing_factor  = 1
    if(0.5 <total_mem_gb < 2):
        dividing_factor = 3
    elif(2< total_mem_gb < 6):
        dividing_factor = 4
    elif( 6 <total_mem_gb < 10):
        dividing_factor = 5
    elif(total_mem_gb >10):
        dividing_factor = 6
    return total_mem_usage,dividing_factor
def save_image_features(img_width, img_height,all_features_hdf5,all_labels_hdf5,train):
    datagen = ImageDataGenerator()
    #Build a fake model
    model = Sequential()
    generator = datagen.flow_from_directory(
        full_data_dir,
        target_size=(img_height,img_width),
        batch_size=batch_size,
        interpolation='lanczos',
        class_mode=None,
        table_pd =train)
   
    shape = (generator.samples, img_width,img_height,3 )
    chunk_shape=(1, 100,100,3)
    total_mem_usage,dividing_factor = calculateDividingFactor(shape)
    f1_all = h5c.File(all_features_hdf5, 'w',chunk_cache_mem_size=total_mem_usage//dividing_factor)
    f1_label = h5.File(all_labels_hdf5, 'w')
    d1_all = f1_all.create_dataset('data', shape ,dtype='float32',chunks=chunk_shape,compression="lzf")
    d1_label = f1_label.create_dataset('data', (generator.samples,len(generator.table_pd.columns)) ,dtype='float32')
    model.write_generator(generator, 
                          steps = int(ceil(generator.samples/ batch_size)),
                           max_queue_size=10,
                           workers=4,
                           use_multiprocessing=False,
                           verbose=0,
                           d_set= d1_all,
                           label_set =d1_label)
    f1_all.close()
    f1_label.close()
img_width, img_height = 224, 224
all_features_hdf5 = '/content/Animals/all_image_features_224.h5'
all_labels_hdf5 = '/content/Animals/all_image_labels_224.h5'
save_image_features(img_width, img_height,all_features_hdf5,all_labels_hdf5,train)
#img_width, img_height = 299, 299
#all_features_hdf5 = '/content/Animals/all_image_features_299.h5'
#all_labels_hdf5 = '/content/Animals/all_image_labels_299.h5'
#save_image_features(img_width, img_height,all_features_hdf5,all_labels_hdf5,train)
#img_width, img_height = 329, 329
#all_features_hdf5 = '/content/Animals/all_image_features_329.h5'
#all_labels_hdf5 = '/content/Animals/all_image_labels_329.h5'
#save_image_features(img_width, img_height,all_features_hdf5,all_labels_hdf5,train)
