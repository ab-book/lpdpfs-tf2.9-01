#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

fig,axes=plt.subplots(2, 5,figsize=(10,4))
axes=axes.flatten()
for i in range(10):
    axes[i].imshow(train_images[i],cmap="gray_r")
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.tight_layout()
plt.show()


# # 将原本28*28 images 转换 a 784 vector
x_train = train_images.reshape(60000,784).astype('float32')
x_test = test_images.reshape(10000,784).astype('float32')
print('x_train:',x_train.shape)
print('x_test:',x_test.shape)

#查看image图像第0项内容
print(x_train[0])



# 将数字图像images的数字标准化，即normalize input from 0-255 to 0-1
X_train = x_train/255
X_test = x_test/255
#查看数字图像images的数字标准化后的结果
print(X_train[0])


print(train_labels[0])

#查看训练数据label标签字段的前5项训练数据
print(train_labels[:5])
#将训练数据和测试数据的类别进行one-hot独热编码
from keras.utils import to_categorical
Y_train = to_categorical(train_labels)
Y_test = to_categorical(test_labels)
#查看进行one-hot encoding转换之后label标签字段的前5项训练数据
print(Y_train[:5])


from tensorflow.keras import models
from tensorflow.keras import layers
# #建立模型
model = models.Sequential()
# #将输入层与隐藏层加入模型，定义隐藏层神经元个数为784
# #设置输入层神经元个数为784
model.add(layers.Dense(units=784,activation='relu',
                        input_dim=784,kernel_initializer='normal'))
# #建立输出层，共10个神经元，对应0到9十个数字，使用激活函数softmax
model.add(layers.Dense(units=10,kernel_initializer='normal',
                         activation='softmax'))
print(model.summary())

# #用compile方法对训练模型进行设置，设置优化器，损失函数
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
# #开始训练
train_history=model.fit(X_train,Y_train,
                          epochs=5,batch_size=200,
                         validation_split=0.2,verbose=2)

# #评价模型
test_loss , test_acc = model.evaluate(X_test,Y_test)
# #输出精度
print('test_acc:',test_acc)
#
#prediction=model.predict_classes(x_test)
prediction=model.predict(x_test)
print('测试数据第340项的真实值:',test_labels[340])
print('测试数据第340项的预测值:',prediction[340])
print('测试数据第341项的真实值:',test_labels[341])
print('测试数据第341项的预测值:',prediction[341])
print('测试数据第342项的真实值:',test_labels[342])
print('测试数据第342项的预测值:',prediction[342])


