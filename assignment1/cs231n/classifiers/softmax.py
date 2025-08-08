from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)

    # TODO: 使用显示循环计算loss和梯度
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train): # 第i个样本
        scores = X[i].dot(W) # 第i个样本的得分向量 (C,)
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores) # 类别概率 (C,)
        loss += -np.log(probs[y[i]]) # 第i个样本的交叉熵损失

        for j in range(num_classes): # 第j个类别
            # X[i]与第j个类别 能贡献到的权重梯度就是 dw[:,j]
            if j == y[i]: # 正确类别的贡献
                dW[:,j] += (probs[j] - 1) * X[i] # dW[:,j] = (exp_scores[j]/np.sum(exp_scores) - 1) * X[i]
            else : # 错误类别的贡献 
                dW[:,j] += probs[j] * X[i] # dW[:,j] = 1/np.sum(exp_scores) * exp_scores[j] * X[i]
        
    loss  = loss / num_train + reg * np.sum(np.square(W))
    dW = dW / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)

    # TODO: 向量化计算loss和梯度
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0] # 样本数N
    scores = X.dot(W)  # (N, C)
    exp_scores = np.exp(scores - np.max(scores))  # (N, C)
    total = np.sum(exp_scores, axis=1,keepdims=True)  # (N, 1)
    probs = exp_scores / total  # (N, C)
    loss = -np.sum(np.log(probs[np.arange(num_train), y])) / num_train + reg * np.sum(np.square(W))
    probs[np.arange(num_train), y] -= 1  # 正确类别减1
    dW = X.T.dot(probs) / num_train + 2 * reg * W # (D, C)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
