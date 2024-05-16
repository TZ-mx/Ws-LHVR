from __future__ import print_function
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
import os
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1.keras.backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == 'channel_first' else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = keras.layers.Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = keras.layers.Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = keras.layers.GlobalAveragePooling1D()(input_feature)
    avg_pool = keras.layers.Reshape(( 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == ( 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == ( 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == ( 1, channel)

    max_pool = keras.layers.GlobalMaxPooling1D()(input_feature)
    max_pool = keras.layers.Reshape(( 1, channel))(max_pool)
    assert max_pool.shape[1:] == ( 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == ( 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == ( 1, channel)

    cbam_feature = keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = keras.layers.Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == 'channel_first':
        cbam_feature = keras.layers.Permute((2, 1 ))(cbam_feature)

    return keras.layers.multiply([input_feature, cbam_feature])



class ExternalAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExternalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.linear = keras.layers.Conv1D(1, 1, padding='same', activation='sigmoid', kernel_initializer='he_normal')
        super(ExternalAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = K.mean(inputs, axis=-1, keepdims=True)
        max_pool = K.max(inputs, axis=-1, keepdims=True)
        concat = keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        weights = self.linear(concat)
        return inputs * weights


def build_WsHVR(input_shape, n_feature_maps, nb_classes):
    x = keras.layers.Input(shape=input_shape)

    conv2 = keras.layers.Conv1D(n_feature_maps * 1, 1, 1, padding='same')(x)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    conv2 = keras.layers.AveragePooling1D(pool_size=2, padding='same')(conv2)

    conv3 = keras.layers.Conv1D(n_feature_maps * 1, 3, 1, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    conv3 = keras.layers.AveragePooling1D(pool_size=2, padding='same')(conv3)


    conv6 = keras.layers.Conv1D(n_feature_maps * 1, 5, 1, padding='same')(conv3)
    conv6 = keras.layers.BatchNormalization()(conv6)
    conv6 = keras.layers.Activation('relu')(conv6)
    conv6 = keras.layers.AveragePooling1D(pool_size=2, padding='same')(conv6)

    attention = channel_attention(conv6, ratio=8)
    attention = ExternalAttention()(attention)
    ARCAM = keras.layers.add([attention, conv6])

    conv4 = keras.layers.Conv1D(n_feature_maps * 1, 5, 1, padding='same')(ARCAM)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Activation('relu')(conv4)
    conv4 = keras.layers.AveragePooling1D(pool_size=2, padding='same')(conv4)



    conv5 = keras.layers.Conv1D(n_feature_maps * 2, 8, 1, padding='same')(conv4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Activation('relu')(conv5)
    conv5 = keras.layers.AveragePooling1D(pool_size=2, padding='same')(conv5)



    output = keras.layers.GlobalAveragePooling1D()(conv5)
    output = keras.layers.Dense(nb_classes, activation='softmax')(output)

    return x, output

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


nb_epochs = 175

flist = ['0.8_2.4_']
for each in flist:
    fname = each
    x_data, y_data = readucr(fname + '.csv')
    nb_classes = len(np.unique(y_data))
    print(nb_classes)
    y_data = (y_data - y_data.min()) / (y_data.max() - y_data.min()) * (nb_classes - 1)
    Y_data = keras.utils.to_categorical(y_data, nb_classes)
    x_data_mean = x_data.mean()
    x_data_std = x_data.std()
    x_data = (x_data - x_data_mean) / (x_data_std)
    x_data = x_data[..., np.newaxis]

(trainX, testX, trainY, testY) = train_test_split(x_data, Y_data, test_size=0.2, random_state=42)

batch_size = min(trainX.shape[0] // 10, 16)

inputs = keras.Input(shape=trainX.shape[1:])
x, y = build_WsHVR((trainX.shape[1:]), 32, nb_classes)

model = keras.models.Model(inputs=x, outputs=y)
optimizer = keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                              patience=50, min_lr=0.0001)
hist = model.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epochs,
                 verbose=1, validation_split=0.1, callbacks=[reduce_lr])
sess = K.get_session()
graph = sess.graph
stats_graph(graph)

start_time = time.time()
scores = model.evaluate(testX, testY, verbose=0)
secs = time.time() - start_time
print(secs)
print(model.metrics_names)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
print("%s: %.2f%%" % (model.metrics_names[3], scores[3] * 100))
accuracy = scores[1]
precision = scores[2]
recall = scores[3]


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'ko', label='Training acc')
plt.plot(epochs, val_acc, 'k', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
#
plt.plot(epochs, loss, 'ko', label='Training loss')
plt.plot(epochs, val_loss, 'k', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
#
plt.show()

