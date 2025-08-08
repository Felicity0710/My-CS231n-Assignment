from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

# W:权重矩阵 (D,C)
# X:测试集 (N,D)
# y:标签 (N,)
# reg:正则化强度
# loss:损失值
# dW:梯度 (D,C)
def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape) # 初始化梯度为0
    num_classes = W.shape[1] # 类别数
    num_train = X.shape[0] # 训练样本数
    loss = 0.0
    for i in range(num_train):
        f=0
        scores = X[i].dot(W) # 计算第i个样本的得分
        correct_class_score = scores[y[i]] # 获取正确类别的得分
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # 这里设置的delta为1
            if margin > 0: # margin大于0才计算损失和梯度，否则损失和梯度都为0
                loss += margin 
                dW[:,j]+=X[i] # 累加梯度
                f+=1 # 计算违反margin的类别数
        dW[:,y[i]] += -f*X[i] # 特别处理正确类别的梯度

    # 平均损失和梯度
    loss /= num_train
    dW/=num_train

    # 添加正则化损失
    loss += reg * np.sum(W * W)

    # TODO: 添加正则化梯度
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW+=2*reg*W
    # 正则化梯度，乘以2是因为我们对W的平方求导

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


# 向量化版本的SVM损失函数
# 参数与返回值意义同上
def svm_loss_vectorized(W, X, y, reg):

    loss = 0.0
    dW = np.zeros(W.shape) 

    # TODO: 向量化计算Loss
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_test=len(X) #样本数
    scores=X.dot(W) # 计算所有样本的得分 (N,C)
    correct_class_scores=scores[np.arange(num_test),y] # 获取所有样本的正确类别得分 (N,)
    margin=np.clip(scores-correct_class_scores.reshape([-1,1])+1,0,None) # 计算margin (N,C)，并将小于0的值设为0
    # correct_class_scores.reshape([-1,1]) 将正确类别得分转换为列向量
    # scores-correct_class_scores.reshape([-1,1]) 计算每个类别的margin
    # np.clip(...,0,None) 将小于0的margin设为0
    margin[np.arange(num_test),y]=0 # 将正确类别的margin设为0
    loss=np.sum(margin)/num_test + reg * np.sum(np.square(W)) # 平均损失加上正则化损失

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # TODO: 向量化计算梯度
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    m = (margin>0).astype(int) # 将margin大于0的部分设为1(表示需要计入这个权重所对应的梯度)，其他部分为0 (N,C)
    f=np.sum(m,axis=1) # 计算每个样本违反margin的类别数 (N,)
    m[np.arange(num_test),y]=-f # 将正确类别的margin设为负的违反类别数 (N,C)，用于特别处理正确类别的梯度
    dW= X.T.dot(m)/num_test + 2*reg*W 
    # X.T.dot(m) 计算每个类别的梯度 (D,C)，dW[i,j]是第i个特征对第j个类别的梯度
    # 那么其就等于X[i].dot(m[:,j])
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
