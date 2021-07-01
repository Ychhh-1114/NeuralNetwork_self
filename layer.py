import numpy as np
from activate_function import *

class Layer:
    def __init__(self,n_input,n_output,activator = Identity,learning_rate = 0.001):
        """
        :param n_input: 输入的个数，即上一层神经元的个数
        :param n_output: 输出的个数，即该层神经元的个数
        :param activator: 激活函数
        :param learning_rate: 学习率

        """
        self.activator = activator
        self.b = np.zeros(n_output).reshape(n_output,1).astype(np.float)    #偏置值
        self.w = np.random.randint(0,1,(n_output,n_input)).astype(np.float) #权值矩阵，为n_output * n_input维
        self.eta = learning_rate

    def forward(self,inputs):
        """
        :param inputs: 上一层各个神经元的输出为 n_input维列向量（正向计算的输入值）
        :return: 无
        """
        self.inputs = inputs   #n_input维
        print((self.w.dot(inputs)).shape)
        print(self.b.shape)

        self.sum = self.w.dot(inputs) + self.b        #对上一层各个神经元的输出求和获得该神经元的输入
        self.outputs = self.activator.value(self.sum) #该层的前向输出
        return self.outputs


    def backward(self,delta_in):
        """
        :param delta_in: 该层的残差(反向计算的输入值)
        :return:
        """
        d = self.activator.derivate(self.sum) * delta_in  #该层的反向激活值,为n_output维
        self.delta_out = self.w.T.dot(d)   #反向计算的输出值

        self.w_grad = d.dot(self.inputs)   #向量外积构造梯度矩阵为n_output * n_input维
        self.b_grad = d                    #b梯度向量

        self.w -= self.eta * self.w_grad
        self.b -= self.eta * d

        return self.delta_out