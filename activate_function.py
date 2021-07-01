import numpy as np



"""
    value:前向激活值
    
    derivate:后向导数值

"""
class ReLU:

    def value(self,z):
        return np.maximum(0,z) #此处不可以用max

    def derivative(self,z):
        return (z > 0).astype(np.int)


class Sigmoid:

    def value(self,z):
        return 1 / (1 + np.exp(-z))


    def derivative(self,z):
        return self.value(z) * (1 - self.value(z))

class Identity:

    def value(self,z):
        return z

    def derivate(self,z):
        return 1