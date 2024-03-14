import tensorflow as tf
import os
import datetime
(train_image,train_labels),(test_image,test_labels)=tf.keras.datasets.mnist.load_data()
train_image.shape,train_labels,test_image.shape,test_labels
train_image = tf.expand_dims(train_image,-1)# 扩充维度
test_image = tf.expand_dims(test_image,-1)# 扩充维度
train_image.shape,test_image.shape
# 改变数据类型
train_image = tf.cast(train_image/255,tf.float32) # 归一化并改变数据类型
train_labels = tf.cast(train_labels,tf.int64)

test_image = tf.cast(test_image/255,tf.float32) # 归一化并改变数据类型
test_labels = tf.cast(test_labels,tf.int64)


train_dataset = tf.data.Dataset.from_tensor_slices((train_image,train_labels)) # 建立数据集
test_dataset = tf.data.Dataset.from_tensor_slices((test_image,test_labels))
train_dataset,test_dataset

train_dataset = train_dataset.repeat().shuffle(60000).batch(128) # 对数据进行洗牌
test_dataset = test_dataset.repeat().batch(128) # 对数据进行洗牌
train_dataset,test_dataset

# 建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,[3,3],activation="relu",input_shape=(None,None,1)),
    tf.keras.layers.Conv2D(32,[3,3],activation="relu"),
    tf.keras.layers.GlobalMaxPool2D(),
    tf.keras.layers.Dense(10,activation="softmax")
])
# 编译模型
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["acc"])

log_dir = os.path.join("logs",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))# 图的存放路径加时间
# 可视化
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1)

# 训练
model.fit(train_dataset,
          epochs=5,
          steps_per_epoch=60000//128,
          validation_data=test_dataset,
          validation_steps=10000//128,
          callbacks=[tensorboard_callback])


# 编译模型
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["acc"])











