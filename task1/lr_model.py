import tensorflow as tf

'''
总结一下：就是把罗辑回归的计算wx + b = y写成代码，最小化交叉熵 
'''
class LrModel(object):
    def __init__(self, config, seq_length):
        self.config = config
        self.seq_length = seq_length
        self.lr()

    def lr(self):
        self.x = tf.placeholder(tf.float32, [None, self.seq_length])
        w = tf.Variable(tf.zeros([self.seq_length, self.config.num_classes]))
        b = tf.Variable(tf.zeros([self.config.num_classes]))

        y = tf.nn.softmax(tf.matmul(self.x, w) + b)

        self.y_pred_cls = tf.argmax(y, 1)#输出行1/列0中最大值的索引，看出来哪个类别概率最大，作为预测结果

        self.y_ = tf.placeholder(tf.float32, [None, self.config.num_classes])
        #函数作用为沿指定轴做平均值计算，默认计算所有元素平均值
        #因为越接近1应该惩罚越小，其中这里的log函数是以e为底的，结合函数图像，越接近1的数值越大，所以需要加负号，惩罚越小
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1]))
        self.loss = tf.reduce_mean(cross_entropy)#下面也是在最小化cross_entropy，那这个loss没有用吗？

        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        #判断是否相等，看有没有预测正确
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
        #cast是转换数据类型，把True和False转换成数字计算预测准确率
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
