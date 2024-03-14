#!/usr/bin/env python
# coding: utf-8


from tensorflow import keras
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 输入图像的尺寸
img_width, img_height = 64,64
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#转成VGG16需要的格式
x_train = [cv2.cvtColor(cv2.resize(i,(img_width, img_height)), 
	cv2.COLOR_GRAY2BGR) for i in x_train]
x_train = np.concatenate([arr[np.newaxis] for arr in 
	x_train]).astype('float32')
x_test  = [cv2.cvtColor(cv2.resize(i,(img_width, img_height)), 
	cv2.COLOR_GRAY2BGR) for i in x_test ]
x_test  = np.concatenate([arr[np.newaxis] for arr in 
	x_test] ).astype('float32')
print(x_train.shape)
print(x_test.shape)




#数据预处理
# 对输入图像归一化
x_train /= 255 
x_test /= 255
# 将输入的标签转换成类别值
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import applications
# weights = "imagenet"：使用imagenet上预训练模型的权重
# 如果weight = None， 则代表随机初始化
# include_top=False：不包括顶层的全连接层
# input_shape：输入图像的维度
conv_base = applications.VGG16(weights = "imagenet", include_top=False, 
	input_shape = (img_width, img_height, 3))



# 我们将已经载入的VGG16的卷积块都固化下来，只训练用于分类的全连接层
for layer in conv_base.layers:
	layer.trainable = False
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Dropout概率0.5
model.add(layers.Dense(10, activation='softmax')) 
print(model.summary())



model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
                batch_size=150,
                epochs=10,
                verbose=2,
                validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])






