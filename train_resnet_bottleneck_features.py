import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Input,Dropout, Flatten, Dense
from keras import applications
from keras.applications.resnet50 import ResNet50, preprocess_input
import keras
from keras import backend as K
import h5py_cache as h5c
import h5py as h5
import random
from sklearn.utils import shuffle
from math import *
import threading
print("Keras Imported")
full_data_dir = '/content/Animals/data/trainval/'
batch_size = 64
img_width, img_height = 224, 224
all_features_hdf5 = '/content/Animals/all_bottleneck_resnet_features_224.h5'
all_labels_hdf5 = '/content/Animals/all_bottleneck_resnet_labels_224.h5'
top_model_weights_path = '/content/Animals/bottleneck_resnet_weights.h5'
#Resnet models should be trained to 30 epochs. Is very hard to tune the parameters end to end.
epochs = 20
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
def save_bottleneck_features(all_features_hdf5,all_labels_hdf5):
    # rescale= 1./255 doest work for Vgg. preprocess_input works
    #datagen = ImageDataGenerator(rescale=1. / 255)
    datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    input_s = Input(shape=(img_width,img_height,3))
    # build the ResNet50 network
    base_model = ResNet50(include_top=False, weights='imagenet',input_tensor=input_s)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    generator = datagen.flow_from_directory(
        full_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        interpolation='lanczos',
        class_mode=None,
        table_pd =train)
    model.write_predict_generator(generator, 
                          steps=generator.samples//batch_size,
                           max_queue_size=10,
                           workers=4,
                           use_multiprocessing=False,
                           verbose=0,
                           h5py_file = all_features_hdf5,
                           h5py_label =all_labels_hdf5)
 

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def train_new_top_model(all_features_hdf5,all_labels_hdf5):
    #The Bottleneck or other image extracted features are  stored in h5 files
    # Use HDF5MatrixCacheIterator to use them. By default HDF5MatrixCacheIterator doesnt transform and doesnt shuffle.
    # so we can use old ImageDataGenerator

    datagen = ImageDataGenerator(validation_split=0.2)
    f1_trainvalidation = h5.File(all_features_hdf5, 'r')
    shape = f1_trainvalidation['data'].shape
    f1_trainvalidation.close()
    #15 Gb is upper limit for cache memory.
    total_mem_usage,dividing_factor = calculateDividingFactor(shape)
    
    #print("Dividing factor {}:".format(dividing_factor))
    print("Cached memory usage: {}".format(total_mem_usage /(1024**3)/dividing_factor))
    chunk_shape = (1, max(shape[1]//2,1), max(shape[2]//2,1),max(shape[3]//2,1))
    f1_trainvalidation = h5c.File(all_features_hdf5, 'r',chunk_cache_mem_size=total_mem_usage//dividing_factor)
    f1_label = h5.File(all_labels_hdf5, 'r')
    train_generator = datagen.flow_hdf5(
        f1_trainvalidation['data'],
        f1_label['data'],
        subset = 'training',
        batch_size=batch_size,
        shuffle=False)
    validation_generator = datagen.flow_hdf5(
        f1_trainvalidation['data'],
        f1_label['data'],
        subset = 'validation',
        batch_size=batch_size,
        shuffle=False)
    model = Sequential()
    model.add(Flatten(input_shape=shape[1:]))
    model.add(Dense(1024, activation='relu'))
    #model.add(Dense(256,input_shape=(train_data_shape[1:],),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(85, activation='sigmoid'))

    #model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',optimizer= "adam", metrics=[f1])
    model.fit_generator(generator = train_generator,
                        steps_per_epoch = int(ceil(train_generator.samples/ batch_size)),
                        validation_data = validation_generator, 
                        max_queue_size=10,  # use a value which can fit batch_size * image_size * max_queue_size in your CPU memory
                        workers=4,  # I don't see multi workers can have any performance benefit without multi threading
                        use_multiprocessing=False,  # HDF5Matrix cannot support multi-threads
                        validation_steps = int(ceil(validation_generator.samples/ batch_size)),
                        shuffle=True,
                        epochs = epochs)
    model.save_weights(top_model_weights_path)
save_bottleneck_features(all_features_hdf5, all_labels_hdf5)
train_new_top_model(all_features_hdf5, all_labels_hdf5)
