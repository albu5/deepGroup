"""
This python script generates resnet features from images of individuals.
These features are then used to predict individual action. (individual_activity_data.m)
"""

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import load_model, Model
from keras.layers import Input
from PIL import Image
from numpy import savetxt, genfromtxt
import numpy as np
import os
from matplotlib import pyplot as plt
import keras.losses
import keras.backend as Keras

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

# Set this path to appropriate directory where collective activity dataset is stored
data_dir = './ActivityDataset'
anno_dir = data_dir + '/' + 'csvanno-posenet'
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# posenet_path = './models-posenet50/total_loss_5'
# pose_resnet = load_model(posenet_path)
# posenet = Model(inputs=pose_resnet.input, outputs=pose_resnet.get_layer('reshape_1').output)


for i in range(1, 45):
    try:
        seq_dir = data_dir + '/' + 'seq%2.2d' % i
        anno_path = anno_dir + '/' + 'data_%2.2d.txt' % i
        anno_data = genfromtxt(anno_path, delimiter=',')
        anno_data_with_resnet = []
        n_frames = np.max(anno_data[:, 0])
        new_tracks = []
        for t in range(1, int(n_frames+1)):
            data_t = anno_data[anno_data[:, 0] == t, :]
            im = Image.open(seq_dir + '/' + 'frame%4.4d.jpg' % t)
            print(seq_dir + '/' + 'frame%4.4d.jpg' % t)
            for j in range(data_t.shape[0]):
                data = data_t[j, :]
                bb = data[2:6]
                region = im.crop((bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]))
                region = region.resize((224, 224), Image.ANTIALIAS)
                im_arr = np.fromstring(region.tobytes(), dtype=np.uint8)
                im_arr = im_arr.reshape((region.size[1], region.size[0], 3))
                img = np.expand_dims(im_arr, axis=0).astype(np.float32)
                img = preprocess_input(img)
                imf = np.squeeze(resnet.predict_on_batch(img))
                row = np.squeeze(data).tolist() + imf.tolist()
                new_tracks.append(row)
        savetxt(anno_dir + '/' + 'resnet_data%2.2d.txt' % i, np.array(new_tracks), delimiter=',')
    except:
        print('skipped this sequence: ', i)

