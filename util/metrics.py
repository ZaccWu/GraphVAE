import tensorflow as tf
import random

class HSIC():
    def __init__(self, zdim):
        self.zdim = zdim

    def __call__(self, z):
        #hsicList = [self.calHSIC(z,i,j) for i in range(self.zdim) for j in range(i)]
        hsicList = [self.calHSIC(z,0,1)]
        self.hsic = tf.convert_to_tensor(hsicList)
        return self.hsic

    def calHSIC(self, z, i, j):
        x = z[:, i]
        y = z[:, j]
        Kx = tf.expand_dims(x, 0) - tf.expand_dims(x, 1)
        Ky = tf.expand_dims(y, 0) - tf.expand_dims(y, 1)
        Kx = tf.exp(-tf.square(Kx))
        Ky = tf.exp(-tf.square(Ky))
        Kxy = tf.matmul(Kx, Ky)
        n = int(Kxy.shape[0])
        h = tf.trace(Kxy) / n ** 2 + tf.reduce_mean(Kx) * tf.reduce_mean(Ky) - 2 * tf.reduce_mean(Kxy) / n
        return h * n ** 2 / (n - 1) ** 2