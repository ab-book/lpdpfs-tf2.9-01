import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# mnist数据集
from tensorflow.keras.datasets import mnist
# Adam优化器
from tensorflow.keras.optimizers import Adam
# 交叉熵损失函数,一般用于多分类
from tensorflow.keras.losses import CategoricalCrossentropy
# 模型和网络层
from tensorflow.keras import Model, layers

# 批次大小
BATCH_SIZE = 128
# 迭代次数
EPOCHS = 10
# 加载mnist的训练、测试数据集
train, test = mnist.load_data()
# 数据集的预处理
@tf.function
def preprocess(x, y):
    # 将x一维数据转为3维灰度图
    x = tf.reshape(x, [28, 28, 1])
    # 将x的范围由[0, 255]为[0, 1]
    x = tf.image.convert_image_dtype(x, tf.float32)
    # 将y数字标签进行独热编码
    y = tf.one_hot(y, 10)
    # 返回处理后的x和y
    return x, y

# 使用Dataset来减少内存的使用
train = tf.data.Dataset.from_tensor_slices(train)
# 对数据进行预处理并且给定BATCH_SIZE
train = train.map(preprocess).batch(BATCH_SIZE)

# test数据集同理
test = tf.data.Dataset.from_tensor_slices(test)
test = test.map(preprocess).batch(BATCH_SIZE)

x = layers.Input(shape=(28, 28, 1))                                                             # 输入为x, 大小为 28*28*1
y = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)   # 64核的卷积层
# 在conv2d和maxpool之间可以使用BatchNormalization来提升训练速度, 可自行百度BatchNormalization的用途
# y = layers.BatchNormalization(axis=3)(y, training=True)  training为True则是训练模式，否则是推理模式
y = layers.MaxPooling2D(pool_size=(2, 2))(y)                                                    # 池化层
y = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(y)  # 128核的卷积层
y = layers.MaxPooling2D(pool_size=(2, 2))(y)
y = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(y)  # 256核的卷积层
y = layers.MaxPooling2D(pool_size=(2, 2))(y)
y = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(y)  # 512核的卷积层
y = layers.MaxPooling2D(pool_size=(2, 2))(y)
y = layers.Flatten()(y)                                                                         # 由于是多维, 于是进行扁平化
y = layers.Dense(10, activation='softmax')(y)  # 10分类, 使用sotfmax激活

# 创建模型
cnn = Model(x, y)
# 打印模型
print(cnn.summary())
# 编译模型,选择优化器、评估标准、损失函数
cnn.compile(optimizer=Adam(learning_rate=1e-4), metrics=['acc'], loss=CategoricalCrossentropy())   # 这里使用初始学习率为1e-4的adam优化器
# 进行模型训练
history = cnn.fit(train, epochs=EPOCHS)
# 测试集的评估
score = cnn.evaluate(test)
# 打印评估成绩
print('loss: {0}, acc: {1}'.format(score[0], score[1]))   # loss: 0.035550699669444456, acc: 0.9883999824523926

# 绘制训练过程中每个epoch的loss和acc的折线图
import matplotlib.pyplot as plt
# history对象中有history字典, 字典中存储着“损失”和“评估标准”
epochs = range(EPOCHS)
fig = plt.figure(figsize=(15, 5), dpi=100)

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(epochs, history.history['loss'])
ax1.set_title('loss graph')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss val')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(epochs, history.history['acc'])
ax2.set_title('acc graph')
ax2.set_xlabel('epochs')
ax2.set_ylabel('acc val')

fig.show()
