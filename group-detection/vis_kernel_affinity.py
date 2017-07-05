
"""
Visualize and save group detections
"""

from utils import read_cad_frames, read_cad_annotations, get_interaction_features, add_annotation, custom_interaction_features
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.merge import add
from keras.optimizers import adam
import keras.backend as kb
from keras.models import load_model
import numpy as np
from scipy import io
from utils import get_group_instance
from matplotlib import pyplot as plt
from keras import losses
from sklearn.cluster import AffinityPropagation, DBSCAN
import os
from numpy import genfromtxt, savetxt


def kernel_loss(y_true, y_pred):
    inclusion_dist = kb.max(y_pred - 1 + y_true)
    exclusion_dist = kb.max(y_pred - y_true)
    exclusion_dist2 = kb.mean(y_pred * (1 - y_true) * kb.cast(y_pred > 0, dtype=kb.floatx()))

    # ex_cost = kb.log(exclusion_dist + kb.epsilon()) * (1 - kb.prod(y_true))
    # in_cost = -kb.log(inclusion_dist + kb.epsilon()) * (1 - kb.prod(1 - y_true))
    ex_cost = (exclusion_dist2 + kb.epsilon()) * (1 - kb.prod(y_true))
    in_cost = -(inclusion_dist + kb.epsilon()) * (1 - kb.prod(1 - y_true))
    # return inclusion_dist * kb.sum(y_true)
    # return - exclusion_dist * (1 - kb.prod(y_true))
    return in_cost + ex_cost


def simple_loss(y_true, y_pred):
    res_diff = (y_true - y_pred) * kb.cast(y_pred >= 0, dtype=kb.floatx())
    return kb.sum(kb.square(res_diff))


'''
======================CONSTANTS==================================================================================
'''
losses.simple_loss = simple_loss
losses.kernel_loss = kernel_loss

if not os.path.exists('res'):
    os.makedirs('res')

model_path = './models/cad-kernel-affinity-bottom-max-long-custom-20.h5'
n_max = 20
cad_dir = '../ActivityDataset'
annotations_dir = cad_dir + '/' + 'csvanno-long-feat'
# annotations_dir = cad_dir + '/' + 'csvanno-long-feat'

annotations_dir_out = cad_dir + '/' + 'csvanno-long-feat-results'
colorstr = ['r', 'g', 'b', 'k', 'w', 'm', 'c', 'y']
n = 11

# specify which sequences are visualized
test_seq = [1, 4, 5, 6, 8, 2, 7, 28, 35, 11, 10, 26]

kernel_net = load_model(model_path)
for n in range(1, 45):
    try:
        if n == 39:
            continue
        f = 1

        pose_vec = genfromtxt('../common/pose/pose%2.2d.txt' % n)
        pose_meta = genfromtxt('../common/pose/meta%2.2d.txt' % n)

        action_vec = genfromtxt('../split1/atomic/actions.txt')
        action_meta = genfromtxt('../split1/atomic/meta.txt')

        if not os.path.exists('res/scene%d' % n):
            os.makedirs('res/scene%d' % n)

        # fig, ax = plt.subplots(1)

        anno_data = read_cad_annotations(annotations_dir, n)
        print(anno_data.shape)
        n_frames = np.max(anno_data[:, 0])

        while True:

            f += 10

            if f > n_frames:
                break
            im = read_cad_frames(cad_dir, n, f)

            bx, by, bp, bi = custom_interaction_features(anno_data, f, max_people=20)
            # print(bx[0].shape, by[0].shape, bp[0].shape)
            # print(len(bx))
            # print(bx[0][:, 18:22])

            anno_data_i = anno_data[anno_data[:, 0] == f, :]
            n_ped = anno_data_i.shape[0]

            affinity_matrix = []

            for j in range(len(bx)):

            	# uncomment this to visualize
                # plt.clf()
                # ax.clear()
                # ax.imshow(im)

                temp = np.squeeze(kernel_net.predict_on_batch(x=[bx[j], bp[j]]))
                affinity_matrix.append(temp[0:n_ped].tolist())

                # uncomment this to visualize individual features
                # print()
                # print(np.round(temp[0:n_ped], 2))
                # print(by[j][0:n_ped, 0])
                # print()
                # add_annotation(ax, bi[j, 2:6], 'k', 2)
                for k in range(n_ped):
                    l = k

                    # uncomment this to visualize individual features
                    # if l is not j:
                        # if np.sum(bi[k, 10:]) > 0:
                            # if temp[l] > 0.5:
                                # add_annotation(ax, bi[k, 2:6], 'b', 2)
                                # ax.arrow(bi[k, 2], bi[k, 3], 64 * bx[k][k, 0], 64 * bx[k][k, 1], fc='b', ec='b',
                                #          head_width=5, head_length=10)
                            # else:
                                # add_annotation(ax, bi[k, 2:6], 'r', 2)
                                # ax.arrow(bi[k, 2], bi[k, 3], 64 * bx[k][k, 0], 64 * bx[k][k, 1], fc='r', ec='r',
                                #          head_width=5, head_length=10)

                # uncomment this to visualize individual features
                # add_annotation(ax, bi[j, 2:6], 'k', 2)
                # ax.arrow(bi[j, 2], bi[j, 3], 64*bx[j][0, 0], 64*bx[j][0, 1], fc='k', ec='k',
                #          head_width=5, head_length=10)
                # print(bi[j, 2], bi[j, 3], 64*bx[j][0, 0], 64*bx[j][0, 1])
                # plt.pause(1./2)
            
            affinity_matrix = np.array(affinity_matrix)
            affinity_matrix[np.isnan(affinity_matrix)] = 0
            # try:
            # print(affinity_matrix)
            if n_ped == 0:
                continue
            af = DBSCAN(eps=0.55, metric='precomputed', min_samples=0, algorithm='auto', n_jobs=1)
            af.fit(1-affinity_matrix)
            # print(af.labels_)
            af_labels = af.labels_
            n_samples = af_labels.shape[0]
            ipm = np.zeros(shape=(n_samples, n_samples))
            for i1 in range(n_samples):
                for i2 in range(n_samples):
                    ipm[i1, i2] = af_labels[i1] == af_labels[i2]
            # print(ipm)
            gt_pm = np.zeros(shape=(n_samples, n_samples))
            for i1 in range(n_samples):
                for i2 in range(n_samples):
                    gt_pm[i1, i2] = by[i1][i2, 0]
            # print(gt_pm)

            # ax.clear()
            # ax.imshow(im)
            # for j in range(len(bx)):
            #     # plt.clf()
            #     add_annotation(ax, bi[j, 2:6], colorstr[af_labels[j]], 2)
            # plt.pause(0.01)
            # plt.savefig('res/scene%d/frame%d.png' % (n, f))
                    ## except:
            #     print('skipped clustering')
            for ped_i in range(af_labels.shape[0]):
                # print(np.sum(np.bitwise_and(anno_data[:, 0] == f, anno_data[:, 1] == ped_i+1)))
                anno_data[np.bitwise_and(anno_data[:, 0] == f, anno_data[:, 1] == ped_i+1), 8] = af_labels[ped_i] + 1

        # save group labels
        savetxt(annotations_dir_out + '/' + 'data_%2.2d.txt' % n, anno_data, delimiter=',')
        print(annotations_dir_out + '/' + 'data_%2.2d.txt' % n)
    except:
        print('skipped', n)

