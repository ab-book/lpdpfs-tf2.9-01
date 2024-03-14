#####MNIST_可视化################################
# 模块导入
import tensorflow as tf
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
