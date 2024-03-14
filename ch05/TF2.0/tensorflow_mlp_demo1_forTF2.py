# 模块导入
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from matplotlib import pyplot as plt

# 导入数据集，分别为输入特征和标签
mnist = tf.keras.datasets.mnist
# (x_train, y_train)：（训练集输入特征，训练集标签)
# (x_test, y_test)：(测试集输入特征，测试集标签)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 可视化训练集输入特征的第一个元素
plt.imshow(x_train[0], cmap="gray")  # 绘制灰度图
plt.show()


# 打印出训练集输入特征的第一个元素
print(f"x_train[0]: \n{x_train[0]}")
# 打印出训练集标签的第一个元素
print(f"y_train[0]: \n {y_train[0]}")

# 打印出整个训练集输入特征的形状
print(f"x_train.shape: \n {x_train.shape}")
# 打印出整个训练集标签的形状
print(f"y_train.shape: \n {y_train.shape}")
# 打印出整个测试集输入特征的形状
print(f"x_test.shape: \n {x_test.shape}")
# 打印出整个测试集标签的形状
print(f"y_test.shape: \n {y_test.shape}")

# 模块导入
import tensorflow as tf

# 导入数据集，分别为输入特征和标签
mnist = tf.keras.datasets.mnist
# (x_train, y_train)：（训练集输入特征，训练集标签)
# (x_test, y_test)：(测试集输入特征，测试集标签)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 对输入网络的输入特征进行归一化，使原本0到255之间的灰度值，变为0到1之间的数值
# （把输入特征的数值变小更适合神经网络吸收）

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

# 搭建网络结构
#构建了一个输入层748，两个128神经元的隐藏，及10个神经元的输出层的神经网络。
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),  # 将输入特征拉直为一维数组，也就是拉直为28*28=784个数值
    tf.keras.layers.Dense(128, activation="relu"),  # 第二层网络128个神经元，使用relu激活函数
    tf.keras.layers.Dense(128, activation="relu"),  # 第三层网络128个神经元，使用relu激活函数
    tf.keras.layers.Dense(10, activation="softmax")  # 输出层网络10个神经元，使用softmax激活函数，使输出符合概率分布
])

# 配置训练方法
model.compile(optimizer="adam",  # 优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 损失函数，输出是概率分布，from_logits=False
              metrics=["sparse_categorical_accuracy"])  # 数据集中的标签是数值，神经网络输出y是概率分布

# 执行训练过程
model.fit(x_train,  # 训练集输入特征
          y_train,  # 训练集标签
          batch_size=32,  # 每次喂入网络32组数据
          epochs=5,  # 数据集迭代5次
          validation_data=(x_test, y_test),  # 测试集输入特征，测试集标签
          validation_freq=1)  # 每迭代1次训练集执行一次测试集的评测

# 打印出网络结构和参数统计
model.summary()

val_loss,val_acc=model.evaluate(x_test,y_test) #测试，获取准确率

predictions=model.predict([x_test[5:8]])#识别测试集中第6-8张图片
#以下是预测，接近1的数据对应的下标就是预测结果。
print(predictions)

#可以使用matplotlib查看一下原始图片，检验一下结果
plt.imshow(x_test[5],cmap="gray")
plt.show()