#!/usr/bin/env python
# coding: utf-8

import os
base_dir = 'cat-and-dog'
#构造路径存储训练数据，校验数据以及测试数据
train_dir = os.path.join(base_dir, 'training_set')
os.makedirs(train_dir, exist_ok = True)
validation_dir = os.path.join(base_dir, 'validation_set')
os.makedirs(validation_dir, exist_ok = True)
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
#我们来检查一下，看看每个分组（训练 / 验证 ）中分别包含多少张图像
print('total trainning cat images: ', len(os.listdir(train_cats_dir)))
print('total trainning dog images: ', len(os.listdir(train_dogs_dir)))
print('total validation cat images: ', len(os.listdir(validation_cats_dir)))
print('total validation dog images: ', len(os.listdir(validation_dogs_dir)))





from tensorflow.keras import layers
from tensorflow.keras import models
model = models.Sequential()
#输入图片大小是150*150 3表示图片像素用(R,G,B)表示
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150 , 150, 3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))





print(model.summary())





from tensorflow.keras import optimizers
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])





from keras.preprocessing.image import ImageDataGenerator
#把像素点的值除以255，使之在0到1之间
train_datagen = ImageDataGenerator(rescale = 1./ 255)
test_datagen = ImageDataGenerator(rescale = 1. / 255)
#generator 实际上是将数据批量读入内存，使得代码能以for in 的方式去方便的访问
# 使用flow_from_directory()方法可以实例化一个针对图像batch的生成器
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150),#  # 将所有图像大小调整为150*150
    batch_size=20,
    class_mode = 'binary')#因为使用了binary_crossentropy损失，所以需要使用二进制标签
validation_generator = test_datagen.flow_from_directory(
    validation_dir,target_size = (150, 150),batch_size = 20,
    class_mode = 'binary')





train_history = model.fit(train_generator, steps_per_epoch = 150,
                             epochs = 30, validation_data = validation_generator,
                                    verbose=2,validation_steps = 100)





try:
    model.save('cats_and_dogs_cnn.h5')
    print('保存模型成功！')
except:
    print('保存模型失败！')





import matplotlib.pyplot as plt
acc = train_history.history['acc']
val_acc = train_history.history['val_acc']
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
epochs = range(1, len(acc) + 1)
#绘制模型对训练数据和校验数据判断的准确率
plt.plot(epochs, acc, 'bo', label = 'trainning acc')
plt.title('Trainning and validation accuary')
plt.legend()
plt.show()
plt.figure()
#绘制模型对训练数据和校验数据判断的错误率
plt.plot(epochs, loss, 'bo', label = 'Trainning loss')
plt.title('Trainning  loss')
plt.legend()
plt.show()






