import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Input,Dropout, Flatten, Dense
from keras import applications,optimizers
from keras.applications.resnet50 import ResNet50, preprocess_input
import keras
from keras import backend as K
import h5py_cache as h5c
import h5py as h5
import random
from sklearn.utils import shuffle
from math import *
import threading
batch_size = 64
img_width, img_height = 224, 224
all_features_hdf5 = '/content/Animals/all_image_features_224.h5'
all_labels_hdf5 = '/content/Animals/all_image_labels_224.h5'
top_model_weights_path = '/content/Animals/bottleneck_resnet_weights.h5'
epochs = 30
def calculateDividingFactor(shape):
    total_mem_usage = min(np.prod(shape)*4,1024**2*15000)
    total_mem_gb = total_mem_usage/(1024**3)
    dividing_factor  = 1
    if(0.5 <total_mem_gb < 2):
        dividing_factor = 2
    elif(2< total_mem_gb < 6):
        dividing_factor = 3
    elif( 6 <total_mem_gb < 10):
        dividing_factor = 4
    elif(total_mem_gb >10):
        dividing_factor = 5
    return total_mem_usage,dividing_factor
  
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

 
input_s = Input(shape=(img_height,img_width,3))
# build the ResNet50 network
base_model = ResNet50(include_top=False, weights='imagenet',input_tensor=input_s)

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(1024, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(85, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base

model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:6]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
#model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=[f1])
# prepare data augmentation configuration

# Raw Features need to be transformed and standardized
datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                            validation_split=0.2,
                            apply_gen_transform= True)

f1_trainvalidation = h5.File(all_features_hdf5, 'r')
shape = f1_trainvalidation['data'].shape
f1_trainvalidation.close()
#15 Gb is upper limit for cache memory.
total_mem_usage,dividing_factor = calculateDividingFactor(shape)

#print("Dividing factor {}:".format(dividing_factor))
print("Cached memory usage: {}".format(total_mem_usage /(1024**3)/dividing_factor))
chunk_shape = (1, 100, 100, 3)
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
    
# fine-tune the model
model.fit_generator(generator = train_generator,
                        steps_per_epoch = int(ceil(train_generator.samples/ batch_size)),
                        validation_data = validation_generator, 
                        max_queue_size=10,  # use a value which can fit batch_size * image_size * max_queue_size in your CPU memory
                        workers=4,  
                        use_multiprocessing=False, 
                        validation_steps = int(ceil(validation_generator.samples/ batch_size)),
                        shuffle=True,
                        epochs = epochs)
