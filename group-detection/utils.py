import numpy as np
import os
from random import choice as choose
from numpy import genfromtxt
from random import randint
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from skimage.io import imread
from matplotlib import patches
from shapely.geometry import Polygon
from matplotlib import pyplot as plt

nex = 1
curr_pos = 0


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_batch(data_dir, batch_size, seq_len):
    global nex
    global curr_pos
    ex_dir = data_dir + '/' + str(nex)
    tracks = np.genfromtxt(ex_dir + '/' + 'tracks.txt', delimiter=',')
    tracks_len = tracks.shape[1]
    x = np.empty(shape=(batch_size, seq_len, 4), dtype=np.float32)
    y = np.empty(shape=(batch_size, seq_len, 2), dtype=np.float32)
    good_ex = []
    for i in range(batch_size):
        if tracks_len < seq_len + curr_pos:
            temp_y = 0
            temp_x = 0
            good_ex.append(False)
            nex = randint(1, 639)
            curr_pos = 0
            ex_dir = data_dir + '/' + str(nex)
            # print(ex_dir)
            tracks = np.genfromtxt(ex_dir + '/' + 'tracks.txt', delimiter=',')
            tracks_len = tracks.shape[1]
        else:
            temp_x = tracks[1:5, curr_pos: curr_pos + seq_len]
            temp_y = np.expand_dims(tracks[5][curr_pos: curr_pos + seq_len], axis=0)
            if np.any(temp_y == 0):
                good_ex.append(False)
            else:
                good_ex.append(True)
            temp_y = np.vstack((temp_y == 1, temp_y == 2))
            curr_pos += seq_len

        if good_ex[-1]:
            x[i, :, :] = np.transpose(temp_x)
            y[i, :, :] = np.transpose(temp_y)
    # print(np.array(good_ex))
    x = x[np.array(good_ex), :, :]
    y = y[np.array(good_ex), :, :]
    return x.astype(np.float32), y.astype(np.float32)


def get_batch_resnet(data_dir, batch_size, seq_len):
    global nex
    global curr_pos
    ex_dir = data_dir + '/' + str(nex)
    tracks = np.genfromtxt(ex_dir + '/' + 'tracks.txt', delimiter=',')
    tracks_len = tracks.shape[1]
    resnet_features = np.genfromtxt(ex_dir + '/' + 'resnet50.txt', delimiter=',')
    x = np.empty(shape=(batch_size, seq_len, resnet_features.shape[1]), dtype=np.float32)
    y = np.empty(shape=(batch_size, seq_len, 2), dtype=np.float32)
    good_ex = []
    for i in range(batch_size):
        if tracks_len < seq_len + curr_pos:
            temp_y = 0
            temp_x = 0
            good_ex.append(False)
            nex = randint(1, 639)
            curr_pos = 0
            ex_dir = data_dir + '/' + str(nex)
            print(ex_dir)
            tracks = np.genfromtxt(ex_dir + '/' + 'tracks.txt', delimiter=',')
            tracks_len = tracks.shape[1]
            resnet_features = np.genfromtxt(ex_dir + '/' + 'resnet50.txt', delimiter=',')
        else:
            temp_x = resnet_features[curr_pos: curr_pos + seq_len, :]
            temp_y = np.expand_dims(tracks[5][curr_pos: curr_pos + seq_len], axis=0)
            if np.any(temp_y == 0):
                good_ex.append(False)
            else:
                good_ex.append(True)
            temp_y = np.vstack((temp_y == 1, temp_y == 2))
            curr_pos += seq_len

        if good_ex[-1]:
            x[i, :, :] = temp_x
            y[i, :, :] = np.transpose(temp_y)
    # print(np.array(good_ex))
    x = x[np.array(good_ex), :, :]
    y = y[np.array(good_ex), :, :]
    return x.astype(np.float32), y.astype(np.float32)


def get_batch_image(data_dir, batch_size, seq_len):
    global nex
    global curr_pos
    ex_dir = data_dir + '/' + str(nex)
    tracks = np.genfromtxt(ex_dir + '/' + 'tracks.txt', delimiter=',')
    tracks_len = tracks.shape[1]
    x = np.empty(shape=(batch_size, seq_len, 224*224*3), dtype=np.float32)
    y = np.empty(shape=(batch_size, seq_len, 2), dtype=np.float32)
    good_ex = []
    for i in range(batch_size):
        if tracks_len < seq_len + curr_pos:
            temp_y = 0
            temp_x = 0
            good_ex.append(False)
            nex = randint(1, 639)
            curr_pos = 0
            ex_dir = data_dir + '/' + str(nex)
            print(ex_dir)
            tracks = np.genfromtxt(ex_dir + '/' + 'tracks.txt', delimiter=',')
            tracks_len = tracks.shape[1]
        else:
            temp_x = np.empty(shape=(seq_len, 224, 224, 3))
            ids = tracks[0, curr_pos: curr_pos + seq_len]
            ctr = 0
            for idx in ids:
                im_path = ex_dir + '/' + '%4.4d.jpg' % idx
                im = image.load_img(im_path, target_size=(224, 224))
                im_ = image.img_to_array(im)
                temp_x[ctr, :, :, :] = im_
                ctr += 1
            temp_x = preprocess_input(temp_x)
            temp_x = np.reshape(temp_x, newshape=(seq_len, 224*224*3))
            temp_y = np.expand_dims(tracks[5][curr_pos: curr_pos + seq_len], axis=0)
            if np.any(temp_y == 0):
                good_ex.append(False)
            else:
                good_ex.append(True)
            temp_y = np.vstack((temp_y == 1, temp_y == 2))
            curr_pos += seq_len

        if good_ex[-1]:
            x[i, :, :] = temp_x
            y[i, :, :] = np.transpose(temp_y)
    # print(np.array(good_ex))
    x = x[np.array(good_ex), :, :]
    y = y[np.array(good_ex), :, :]
    return x.astype(np.float32), y.astype(np.float32)


def get_image_names_and_labels(data_dir):
    labels_ind = genfromtxt(os.path.join(data_dir, 'labels.txt'))-1
    labels = np.zeros((labels_ind.shape[0], 8))
    labels[np.arange(labels_ind.shape[0]).astype(np.int), labels_ind.astype(np.int)] = 1
    img_list = []
    for i in range(labels.shape[0]):
        img_list.append(os.path.join(data_dir, '%d.jpg' % (i + 1)))
    return img_list, labels


def get_parse_batch(batch_size, img_list, labels, height, width, n_classes=8):
    n_files = len(img_list)
    batch_images = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size, n_classes), dtype=np.float32)

    # good_ids = []
    for i in range(batch_size):
        idx = randint(0, n_files - 1)
        img = image.load_img(img_list[idx], target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        labels_i = labels[idx, :]
        batch_images[i, :, :, :] = img
        batch_labels[i, :] = labels_i
    return batch_images, batch_labels


def get_parse_batch_test(img_list, labels, height, width, start_i, end_i, n_classes=8):
    n_files = len(img_list)
    batch_images = np.zeros((end_i-start_i, height, width, 3), dtype=np.float32)
    batch_labels = np.zeros((end_i-start_i, n_classes), dtype=np.float32)

    # good_ids = []
    i = start_i
    while i < end_i:
        idx = i
        img = image.load_img(img_list[idx], target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        labels_i = labels[idx, :]
        batch_images[i-start_i, :, :, :] = img
        batch_labels[i-start_i, :] = labels_i
        i += 1
    return batch_images, batch_labels


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum(np.dot(v, b) * b for b in basis)
        if (w > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)


def get_group_instance(gd):
    n_instances = gd.shape[0]
    choice = np.random.randint(0, n_instances-1)
    inst = gd[choice]
    persons = inst[0]
    features = []
    labels = []
    inst_dict = {}
    for person in persons:
        features.append(person['features'][0].tolist())
        labels.append(person['group'][0][0])
        inst_dict = {'features': np.array(features), 'labels': np.array(labels)}
    return inst_dict


def get_all_group_instances(gd):
    instances = []
    for inst in gd:
        persons = inst[0]
        features = []
        labels = []
        for person in persons:
            features.append(person['features'][0].tolist())
            labels.append(person['group'][0][0])
        inst_dict = {'features': np.array(features), 'labels': np.array(labels)}
        instances.append(inst_dict)


def read_cad_frames(data_dir, seqi, framei):
    seq_dir = data_dir + '/' + 'seq%2.2d' % seqi
    im_path = seq_dir + '/' + 'frame%4.4d' % framei + '.jpg'
    return imread(im_path)


def read_cad_annotations(anno_dir, seqi):
    anno_path = anno_dir + '/' + 'data_%2.2d' % seqi + '.txt'
    return genfromtxt(anno_path, delimiter=',')


def add_annotation(plt_axes, bbs, colour_str='r', line_width=1):
    rect = patches.Rectangle((bbs[0], bbs[1]), bbs[2], bbs[3], linewidth=line_width, edgecolor=colour_str,
                             facecolor='none')
    plt_axes.add_patch(rect)
    # return plt_axes


def get_interaction_features(annotation_data, frame_i, max_people=10):
    frame_i_idx = annotation_data[:, 0] == frame_i
    annotation_data_i = annotation_data[frame_i_idx, :]
    batch_x = []
    batch_y = []
    batch_pad = []
    for member in range(annotation_data_i.shape[0]):
        pair_feat = []
        pair_membership = []
        pair_pad = []
        for agent in range(annotation_data_i.shape[0]):
            if agent is not member:
                pair_pad.append(0)
            else:
                pair_pad.append(-2)
            pair_feat.append(np.hstack((annotation_data_i[member, 10:], annotation_data_i[agent, 10:])).tolist())
            pair_membership.append(annotation_data_i[member, 8] == annotation_data_i[agent, 8])

        n_remaining = max_people - len(pair_membership)
        for dummy in range(n_remaining):
            pair_feat.append(pair_feat[0])
            pair_membership.append(pair_membership[0])
            pair_pad.append(-2)

        batch_x.append(np.array(pair_feat).astype(np.float32))
        batch_y.append(np.expand_dims(np.array(pair_membership).astype(np.float32), axis=1))
        batch_pad.append(np.expand_dims(np.array(pair_pad).astype(np.float32), axis=1))
    return batch_x, batch_y, batch_pad, annotation_data_i


def get_feature_vector(desc1, desc2):
    feat_vec = []
    v1 = desc1[0:4]
    v2 = desc2[0:4]
    w1 = desc1[8]
    h1 = desc1[9]
    w2 = desc2[8]
    h2 = desc2[9]
    s1 = np.sqrt(w1 * h1)
    s2 = np.sqrt(w2 * h2)
    x1 = desc1[6:8]
    x2 = desc2[6:8]
    action1 = desc1[5]
    action2 = desc2[5]
    feat_vec.append(np.abs(v1[0] - v2[0]))
    feat_vec.append(np.abs(v1[1] - v2[1]))
    feat_vec.append(np.abs(v1[2] - v2[2]))
    feat_vec.append(np.abs(v1[3] - v2[3]))
    feat_vec.append(np.abs(v1[0] - v2[0]) / np.sqrt(s1 * s2))
    feat_vec.append(np.abs(v1[1] - v2[1]) / np.sqrt(s1 * s2))
    feat_vec.append(np.abs(v1[2] - v2[2]) / np.sqrt(s1 * s2))
    feat_vec.append(np.abs(v1[3] - v2[3]) / np.sqrt(s1 * s2))
    feat_vec.append(np.abs(w1 - w2))
    feat_vec.append(np.abs(h1 - h2))
    feat_vec.append(np.abs(x1[0] - x2[0]))
    feat_vec.append(np.abs(x1[1] - x2[1]))
    tx1 = desc1[10:13]
    ty1 = desc1[13:16]
    tx2 = desc2[10:13]
    ty2 = desc2[13:16]

    xy1 = [(tx1[i], ty1[i]) for i in range(tx1.shape[0])]
    xy2 = [(tx2[i], ty2[i]) for i in range(tx2.shape[0])]
    poly1 = Polygon(xy1)
    poly2 = Polygon(xy2)
    poly_i = poly1.intersection(poly2)
    ax = plt.subplot(111)
    plot_coords(ax, poly1)
    plot_bounds(ax, poly1)
    plot_coords(ax, poly2)
    plot_bounds(ax, poly2)
    plot_coords(ax, poly_i)
    plot_bounds(ax, poly_i)
    plt.show()
    feat_vec.append(poly_i.area)
    feat_vec.append(np.abs(action1-action2))

    return feat_vec


def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)


def plot_bounds(ax, ob):
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, 'o', color='#000000', zorder=1)


def custom_interaction_features(annotation_data, frame_i, max_people=10):
    frame_i_idx = annotation_data[:, 0] == frame_i
    annotation_data_i = annotation_data[frame_i_idx, :]
    batch_x = []
    batch_y = []
    batch_pad = []
    for member in range(annotation_data_i.shape[0]):
        pair_feat = []
        pair_membership = []
        pair_pad = []
        for agent in range(annotation_data_i.shape[0]):
            if agent is not member:
                pair_pad.append(0)
            else:
                pair_pad.append(-2)
            pair_feat.append(get_feature_vector(annotation_data_i[member, 10:], annotation_data_i[agent, 10:]))
            pair_membership.append(annotation_data_i[member, 8] == annotation_data_i[agent, 8])

        n_remaining = max_people - len(pair_membership)
        for dummy in range(n_remaining):
            pair_feat.append(pair_feat[0])
            pair_membership.append(pair_membership[0])
            pair_pad.append(-2)

        batch_x.append(np.array(pair_feat).astype(np.float32))
        batch_y.append(np.expand_dims(np.array(pair_membership).astype(np.float32), axis=1))
        batch_pad.append(np.expand_dims(np.array(pair_pad).astype(np.float32), axis=1))
    return batch_x, batch_y, batch_pad, annotation_data_i
