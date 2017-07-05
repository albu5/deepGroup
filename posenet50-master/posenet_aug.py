from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Reshape, Dropout
from keras.models import Model, load_model
from keras.optimizers import adam, rmsprop
import os
from numpy import genfromtxt
from matplotlib import pyplot as plt
from utils import get_image_names_and_labels, get_parse_batch
from random import randint
from keras import backend as Keras
import numpy as np
import keras


def preprocess_input3(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = Keras.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x

'''
========================================================================================================================
CONSTANTS
========================================================================================================================
'''
batch_size = 32
num_iter = 10000000
decay_step = 10000000
save_step = 1000
disp_step = 10
eval_step = 10
model_path = './models-posenet50/total_loss_all_aug'


'''
========================================================================================================================
CUSTOM LOSSES HERE
========================================================================================================================
'''


def good_loss(y_true, y_pred):
    return Keras.mean(-y_true * Keras.log(y_pred + Keras.epsilon()), axis=-1)


def bad_loss(y_true, y_pred):
    cost_matrix_np = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                               [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                               [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                               ], dtype=np.float32)
    cost_matrix = Keras.constant(value=cost_matrix_np, dtype=Keras.floatx())
    bad_pred = Keras.dot(y_true, cost_matrix)
    return Keras.mean(-bad_pred * Keras.log(1 - y_pred + Keras.epsilon()), axis=-1)


def total_loss(y_true, y_pred):
    return bad_loss(y_true, y_pred) + good_loss(y_true, y_pred)

keras.losses.total_loss = total_loss
'''
========================================================================================================================
MODEL DEFINITION
========================================================================================================================
'''
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

input_layer = Input(shape=(224, 224, 3))
resnet_features = resnet(input_layer)
resnet_features = Reshape(target_shape=(2048, ))(resnet_features)
# resnet_features = Dropout(0.2)(resnet_features)
resnet_dense = Dense(1024, activation='relu')(resnet_features)
resnet_prob = Dense(8, activation='softmax')(resnet_dense)
pose_resnet = Model(inputs=input_layer, outputs=resnet_prob)

optimizer = rmsprop(lr=0.0001)
pose_resnet.compile(optimizer=optimizer, loss=total_loss, metrics=['accuracy'])
print(pose_resnet.summary())

# pose_resnet = load_model(model_path)
'''
========================================================================================================================
TRAINING
========================================================================================================================
'''
train_datagen = ImageDataGenerator(shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=False,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   preprocessing_function=preprocess_input3
                                   )

# train_datagen = ImageDataGenerator(horizontal_flip=False,
#                                    preprocessing_function=preprocess_input3
#                                    )

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input3)

train_generator = train_datagen.flow_from_directory('./../tf-vgg/PARSE224/subdir-test/train',
                                                    target_size=(224, 224),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory('./../tf-vgg/PARSE224/subdir-test/test',
                                                        target_size=(224, 224),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

i = 0
v_acc = 0
while i < 30:
    if i % 2 == 0:
        for layer in resnet.layers:
            layer.trainable = True
            keras.backend.set_value(pose_resnet.optimizer.lr, 0.00001)
    else:
        for layer in resnet.layers:
            layer.trainable = False
            keras.backend.set_value(pose_resnet.optimizer.lr, 0.0001)

    history = pose_resnet.fit_generator(generator=train_generator,
                                        steps_per_epoch=int(1063/1),
                                        epochs=i+1,
                                        validation_data=validation_generator,
                                        validation_steps=int(210/1),
                                        initial_epoch=i)
    temp = pose_resnet.evaluate_generator(validation_generator, int(210/1))
    if temp[1] > v_acc:
        pose_resnet.save(model_path)
        v_acc = temp[1]
        print('model saved...')
    i += 1
