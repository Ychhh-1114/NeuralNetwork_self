import numpy as np
from layer import *
from loss import *
from activate_function import *

class NeuralNetwork:

    def __init__(self,layers,loss,EPOCH = 5000):
        """
        :param layers: 神经网络所有层
        :param loss: 神经网络损失函数
        :param EPOCH: 神经网络训练次数
        """
        self.layers = layers
        self.loss = loss
        self.EPOCH = EPOCH
        np.random.seed(100)  #便于使用SGD并观察


    def forward(self,x): #网络正向传播
        """
        :param x: 神经网络的输入值，即样本的特征
        :return: 神经网络正向输出值
        """
        input = x
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return input

    def backward(self,y,output):
        """
        :param y:真实值
        :param output: 神经网络输出值
        :return:

        通过字符串切片方法，反向进行layer的遍历
        """
        delta_in = self.loss.derivate(y,output)
        for layer in self.layers[::-1]:
            back_out = layer.backward(delta_in)
            delta_in = back_out

    def fit(self,X,y):
        """
        采用SGD进行训练
        :param X: 特征集
        :param y: 特征标签
        :return:
        """
        m,n = X.shape
        for epoch in range(self.EPOCH):
            if epoch % 5000 == 0:
                print("traing ",epoch," times")
            index = np.random.randint(m)
            x_ =  X[index].reshape(-1,1)
            y_ = y[index]
            nn_out = self.forward(x_)
            self.backward(y_,nn_out)

    def predict(self,X):
        m,_ = X.shape
        pred_y = []
        for index,x in enumerate(X):
            input = x.reshape(-1,1)
            out = self.forward(input).flatten()
            pred_y.append(out)

        return np.array(pred_y)

    class Layer:
        def __init__(self, n_input, n_output, activator=Identity, learning_rate=0.001):
            """
            :param n_input: 输入的个数，即上一层神经元的个数
            :param n_output: 输出的个数，即该层神经元的个数
            :param activator: 激活函数
            :param learning_rate: 学习率

            """
            self.activator = activator
            self.b = np.zeros(n_output).reshape(n_output, 1)  # 偏置值
            r = np.sqrt(6 / (n_input + n_output))  #初值须有正有负，且不能太大（参数一多容易出现inf）
            self.w = np.random.uniform(-r, r, (n_output, n_input))  # 权值矩阵，为n_output * n_input维
            self.eta = learning_rate

        def forward(self, inputs):
            """
            :param inputs: 上一层各个神经元的输出为 n_input维列向量（正向计算的输入值）
            :return: 无
            """
            self.inputs = inputs  # n_input维
            self.sum = self.w.dot(inputs) + self.b  # 对上一层各个神经元的输出求和获得该神经元的输入
            self.outputs = self.activator.value(z = self.sum)  # 该层的前向输出
            return self.outputs

        def backward(self, delta_in):
            """
            :param delta_in: 该层的残差(反向计算的输入值)
            :return:反向输出值
            """
            sigma =  np.array((self.activator.derivative(self.sum))).reshape(-1,1)
            d = (sigma * (np.array(delta_in)).reshape(-1,1)).reshape(-1,1)  # 该层的反向激活值,为n_output维
            self.delta_out = self.w.T.dot(d)  # 反向计算的输出值

            self.w_grad = d.dot(self.inputs.T).astype('float64')  # 向量外积构造梯度矩阵为n_output * n_input维
            self.b_grad = d  # b梯度向量

            self.w -= (self.eta * self.w_grad)
            self.b -= (self.eta * d).astype(np.float)

            return self.delta_out

