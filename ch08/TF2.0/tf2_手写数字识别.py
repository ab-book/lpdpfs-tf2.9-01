import tensorflow as tf

import numpy as np
import math
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential
from tensorflow.keras.callbacks import TensorBoard
import datetime
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei','Songti SC','STFangsong']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


#导入数据
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
print(x_train.shape,x_test.shape)

#预处理---正规化
def normalize(x,y):
    x=tf.cast(x,tf.float32)
    x/=255
    return x,y

#添加一层维度，方便后续扁平化
x_train=tf.expand_dims(x_train,axis=-1)
x_test=tf.expand_dims(x_test,axis=-1)

train_dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train))
test_dataset=tf.data.Dataset.from_tensor_slices((x_test,y_test))
train_dataset=train_dataset.map(normalize)
test_dataset=test_dataset.map(normalize)


#画图
plt.figure(figsize=(10,15))
i=0
for (x_test,y_test) in test_dataset.take(25):
    x_test=x_test.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.imshow(x_test,cmap=plt.cm.binary)
    plt.xlabel([y_test.numpy()],fontsize=10)
    i+=1
plt.show()

#开始定义模型
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(64,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.summary()

# 模型编译
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#开始训练
batch_size=32
train_dataset=train_dataset.repeat().shuffle(60000).batch(batch_size)
test_dataset=test_dataset.batch(batch_size)
#为tensorboard可视化保存数据
tensorboard_callback=tf.keras.callbacks.TensorBoard(histogram_freq=1)
model.fit(train_dataset,epochs=5,steps_per_epoch=math.ceil(60000/batch_size),callbacks=[tensorboard_callback])

#模型评估
test_loss,test_accuracy=model.evaluate(test_dataset,steps=math.ceil(10000/32))
print('Accuracy on test_dataset',test_accuracy)

# 模型预测
predictions = model.predict(test_dataset)


# 查看预测结果
def plot_test(i, predictions_array, true_labels, images):
    predic, label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.imshow(img[..., 0], cmap=plt.cm.binary)
    predic_label = np.argmax(predic)
    if (predic_label == label):
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("预测标签为:{},概率:{:2.0f}% (真实标签:{})".format(predic_label, 100 * np.max(predic), label), color=color)



def plot_value(i, predictions_array, true_label):
    predic, label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    thisplot = plt.bar(range(10), predic, color='#777777')
    plt.ylim([0, 1])
    predic_label = np.argmax(predic)
    thisplot[predic_label].set_color('blue')
    thisplot[label].set_color('green')


rows, cols = 5, 3
num_images = rows * cols
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()

plt.figure(figsize=(2 * 2 * cols, 2 * rows))
for i in range(num_images):
    plt.subplot(rows, 2 * cols, 2 * i + 1)
    plot_test(i, predictions, test_labels, test_images)
    plt.subplot(rows, 2 * cols, 2 * i + 2)
    plot_value(i, predictions, test_labels)
plt.show()
