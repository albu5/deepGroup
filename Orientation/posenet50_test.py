from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import load_model
from keras import backend as Keras
import numpy as np
import keras
from keras.preprocessing import image

'''
========================================================================================================================
CONSTANTS
========================================================================================================================
'''
batch_size = 32
model_path = '/total_loss_all_aug'  # change this path to yor model file location


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

keras.losses.good_loss = good_loss
keras.losses.total_loss = total_loss

'''
========================================================================================================================
TESTING CODE
========================================================================================================================
'''


def get_pose_batch_test(img_list, height, width, n_classes=8):
    n_files = len(img_list)
    batch_images = np.zeros((n_files, height, width, 3), dtype=np.float32)

    # good_ids = []
    for i in range(n_files):
        img = image.load_img(img_list[i], target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        batch_images[i, :, :, :] = img
    return batch_images

pose_resnet = load_model(model_path)

# change this list to the image files you want to test
img_paths = ['some_guy.jpg',]

X = get_pose_batch_test(img_paths, 224, 224)
Y = pose_resnet.predict(X, batch_size=batch_size)
print(Y)
