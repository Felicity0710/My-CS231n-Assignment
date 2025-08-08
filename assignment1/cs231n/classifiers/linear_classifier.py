from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    # X: 训练数据 (N, D)
    # y: 标签 (N,)
    # learning_rate: 学习率
    # reg: 正则化强度
    # num_iters: 迭代次数
    # batch_size: batch大小
    # verbose: 是否打印训练过程
    # 输出：每次iteration的loss
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # 类别数
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes) #随机初始化W

        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # TODO: 随机采样一个batch
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            idx=np.random.choice(num_train,batch_size) #从num_train个样本中随机选择batch_size个样本的索引
            X_batch=X[idx,:]
            y_batch=y[idx].reshape(1,-1) #重建为(N,)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # TODO: 更新权重W
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.W-=learning_rate*grad

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    # X: 测试数据 (N, D)
    # 返回预测标签 (N,)
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        # TODO: 计算预测标签
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        y_pred=np.argmax(X.dot(self.W),axis=1).reshape(1,-1)
        # 找到得分最高的样本对应的索引，并重建为列向量

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred

    # 纯虚接口
    def loss(self, X_batch, y_batch, reg):
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
