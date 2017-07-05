from keras.layers import Input, LSTM, Dense, Masking, merge, Dropout
from keras.models import Model, load_model
from keras.optimizers import adam
from keras.utils import to_categorical
import numpy as np
from numpy import genfromtxt, savetxt
from matplotlib import pyplot as plt


data_dir = './split2/scene_activity_data_test_none'
model_path = './scenelstm/scenelstm-split1.h5'
batch_size = 128

trainX1 = (genfromtxt(data_dir + '/' + 'trainX1.csv', delimiter=','))
trainX2 = np.reshape(genfromtxt(data_dir + '/' + 'trainX2.csv', delimiter=','), newshape=(-1, 10, 2048))
trainY = to_categorical(genfromtxt(data_dir + '/' + 'trainY.csv', delimiter=',') - 1)
testX1 = (genfromtxt(data_dir + '/' + 'testX1.csv', delimiter=','))
testX2 = np.reshape(genfromtxt(data_dir + '/' + 'testX2.csv', delimiter=','), newshape=(-1, 10, 2048))
testY = to_categorical(genfromtxt(data_dir + '/' + 'testY.csv', delimiter=',') - 1)

print(trainX1.shape, trainX2.shape, trainY.shape, testX1.shape, testX2.shape, testY.shape)


# freq_layer = Input(shape=(4,))
# context_layer = Input(shape=(10, 2048))
# masked = Masking()(context_layer)
# lstm1 = LSTM(256, activation='sigmoid', recurrent_activation='tanh')(masked)
# drop1 = Dropout(rate=0.95)(lstm1)
# fc_context = Dense(16, activation='tanh')(drop1)
# fc_freq = Dense(16, activation='tanh')(freq_layer)
# merged = merge(inputs=[fc_context, fc_freq], mode='concat', concat_axis=1)
# fc2 = Dense(5, activation='softmax')(merged)
# scene_net = Model(inputs=[freq_layer, context_layer], outputs=fc2)
# print(scene_net.summary())
#
# optm = adam(lr=0.001)
# scene_net.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
scene_net = load_model(model_path)

print(testY.shape, trainY.shape)
scores = scene_net.evaluate(x=[testX1, testX2], y=testY, batch_size=batch_size)
y_fit = scene_net.predict(x=[testX1, testX2])
savetxt(data_dir + '/' + 'scene_res_none.txt', np.hstack((y_fit, testY)))
print(scores)
