import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#首先，要特别注意的是如果出现[Could not load dynamic library cudart64_101.dll]报错只针对GPU版的tessorflow,CPU版的可忽略，
x = tf.compat.v1.Variable([1, 2])
a = tf.compat.v1.constant([3, 3])

sub = tf.compat.v1.subtract(x, a)  # 增加一个减法op
add = tf.compat.v1.add(x, sub)  # 增加一个加法op

# 注意变量再使用之前要再sess中做初始化，但是下边这种初始化方法不会指定变量的初始化顺序
# TensorFlow2.0及以上版本没有global_variables_initializer这个属性
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))


# 创建一个名字为‘counter’的变量 初始化0
state = tf.compat.v1.Variable(0, name='counter')
new_value = tf.compat.v1.add(state, 1)  # 创建一个op，作用是使state加1
update = tf.compat.v1.assign(state, new_value)  # 赋值op,不能直接用等号赋值，作用是 state = new_value，借助tf.assign()函数
init = tf.compat.v1.global_variables_initializer()#全局变量初始化

with tf.compat.v1.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):#循环5次
        sess.run(update)
        print(sess.run(state))