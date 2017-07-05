from keras.layers import Input, LSTM, Dense, Masking
from keras.models import Model, load_model
from keras.optimizers import adam
from keras.utils import to_categorical
import numpy as np
from numpy import genfromtxt, savetxt
from matplotlib import pyplot as plt


data_dir = './split1/group_activity_data_action'
model_path = './actlstm/actlstmfc2-split1.h5'

trainX = np.reshape(genfromtxt(data_dir + '/' + 'trainX.csv', delimiter=','), newshape=(-1, 10, 110))
trainY = to_categorical(genfromtxt(data_dir + '/' + 'trainY.csv', delimiter=',') - 1)
testX = np.reshape(genfromtxt(data_dir + '/' + 'testX.csv', delimiter=','), newshape=(-1, 10, 110))
testY = to_categorical(genfromtxt(data_dir + '/' + 'testY.csv', delimiter=',') - 1)
# testMeta = genfromtxt(data_dir + '/' + 'testMeta.csv', delimiter=',')


# print(trainX.shape, trainY.shape, testX.shape, testY.shape, testMeta.shape)


# model
#
# input_layer = Input(shape=(10, 110))
# masked_in = Masking()(input_layer)
# lstm1 = LSTM(512, activation='sigmoid', recurrent_activation='tanh',
#              return_sequences=False)(masked_in)
# fc1 = Dense(128, activation='tanh')(lstm1)
# fc2 = Dense(4, activation='softmax')(fc1)
# act_lstm = Model(inputs=input_layer, outputs=fc2)
#
# print(act_lstm.summary())
# optimizer = adam(lr=0.0001)
# act_lstm.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
act_lstm = load_model(model_path)


scores = act_lstm.evaluate(testX, testY, batch_size=2048)
y_fit = act_lstm.predict(testX, batch_size=2048)

print(y_fit.shape)

# testMeta = genfromtxt(data_dir + '/' + 'testMeta.csv', delimiter=',')
savetxt(data_dir + '/' + 'testResults_action.csv', y_fit, delimiter=',')

print(scores)

