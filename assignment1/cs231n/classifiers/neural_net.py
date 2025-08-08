from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # X: 测试集 (N,D)
    # y: 标签 (N,)
    # reg: 正则化强度
    # 返回值：loss, grads
    def loss(self, X, y=None, reg=0.0):
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        # TODO: 计算前向传播
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        z1=X.dot(W1) + b1 # 计算第一层的线性变换
        a1=np.maximum(0,z1) # ReLU激活函数
        scores=a1.dot(W2) + b2 # 计算第二层的线性变换

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        # TODO: 计算loss, 包括数据损失和W1,W2的Lw正则化损失， 使用softmax损失函数
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        exp_socres = np.exp(scores - np.max(scores, axis=1, keepdims=True)) # exp得分
        probs = exp_socres / np.sum(exp_socres, axis=1, keepdims=True) # 概率
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N + reg * (np.sum(np.square(W1)) + np.sum(np.square(W2))) # 数据损失 + L2正则化损失

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        # TODO: 计算梯度
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        probs[np.arange(N), y] -= 1 # 正确类别的概率减1
        grads['W2'] = a1.T.dot(probs) / N + 2 * reg * W2
        grads['b2'] = np.sum(probs, axis=0) / N
        da1 = probs.dot(W2.T)
        da1[z1 <= 0] = 0  # ReLU 反向传播，小于等于0的部分梯度为0
        grads['W1'] = X.T.dot(da1) / N + 2 * reg * W1
        grads['b1'] = np.sum(da1, axis=0) / N
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
      
      num_train = X.shape[0]
      iterations_per_epoch = max(num_train / batch_size, 1)

      # Use SGD to optimize the parameters in self.model
      loss_history = []
      train_acc_history = []
      val_acc_history = []

      for it in range(num_iters):
          X_batch = None
          y_batch = None

          # TODO: 创建随机batch
          # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

          idx = np.random.choice(num_train, batch_size)
          X_batch = X[idx,:]
          y_batch = y[idx] 

          # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

          # Compute loss and gradients using the current minibatch
          loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
          loss_history.append(loss)

          # TODO: 梯度下降
          # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

          self.params['W1'] -= learning_rate*grads['W1']
          self.params['b1'] -= learning_rate*grads['b1']
          self.params['W2'] -= learning_rate*grads['W2']
          self.params['b2'] -= learning_rate*grads['b2']


          # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

          #if verbose and it % 100 == 0:
              #print('iteration %d / %d: loss %f' % (it, num_iters, loss))

          # Every epoch, check train and val accuracy and decay learning rate.
          if it % iterations_per_epoch == 0:
              # Check accuracy
              train_acc = (self.predict(X_batch) == y_batch).mean()
              val_acc = (self.predict(X_val) == y_val).mean()
              train_acc_history.append(train_acc)
              val_acc_history.append(val_acc)

              # Decay learning rate
              learning_rate *= learning_rate_decay

      return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
      }

    def predict(self, X):

        y_pred = None

        # TODO: 预测
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        z1=np.dot(X,self.params['W1'])+self.params['b1']
        a1=np.maximum(0,z1)
        scores=np.dot(a1,self.params['W2'])+self.params['b2']
        y_pred=np.argmax(scores,axis=1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
