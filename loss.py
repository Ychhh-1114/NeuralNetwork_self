import numpy as np

class Loss:

    class MSE:

        def value(self, y, v):
            """
            :param y: 真实标签
            :param v: 输出向量
            :return: 均方误差值
            """
            return (np.array(y).flatten() - np.array(v).flatten()) ** 2

        def derivate(self, y, v):
            return 2 * (v - y)

    class SoftmaxCrossEntroy:

        def value(self, y, v):
            p = self.softmax(v)
            return -(y * np.log(p)).sum()

        def derivate(self, y, v):
            p = self.softmax(v.reshape(1,-1))
            # print(p.shape,y.shape)
            return p - y

        def softmax(self,x):
            """
            :param x: 向量
            :return:
            """
            m,n = x.shape
            e = np.exp(x)
            for i in range(m):
                sum = np.sum(e[i].flatten())
                e[i] = e[i] / sum
            return e.reshape(-1,10)