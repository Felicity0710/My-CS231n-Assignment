from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                dists[i,j]=np.sqrt(np.sum(np.square(X[i,:]-self.X_train[j,:])))

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
    # X:测试集
    #使用一个循环
    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #TODO:同上
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            dists[i,:]=np.sqrt(np.sum(np.square(self.X_train-X[i,:]),axis=1))
            # 由于自动扩展，self.X_train的每行j都减去X[i,:]
            # np.square对每个元素平方
            # np.sum(axis=1)对每行元素求和
            # np.sqrt对每个元素开平方
            
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    # X:测试集
    # 不使用循环
    def compute_distances_no_loops(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dists=np.sqrt(np.sum(np.square(self.X_train-X[:,None,:]),axis=2))
        # X[:,None,:]将X的形状从(num_test,D)变为(num_test,1,D)，
        # 这样可以实现自动扩展，X_train的每行都减去X,得到结果的size为(num_test,num_train,D)
        # np.square对每个元素平方
        # np.sum(axis=2)对D所对应的维度的元素求和
        # np.sqrt对每个元素开平方

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    # dists:距离矩阵
    # k:最近邻个数
    # 返回预测标签
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            # TODO: 找到k个最近邻的标签
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            knn=np.argsort(dists[i,:])[0:k] 
            # argsort返回从小到大排序的索引
            # 取前k个索引
            # dists[i,:]是第i个测试样本到所有训练样本的距离
            # knn是第i个测试样本的k个最近邻的索引
            closest_y=self.y_train[knn]
            # 通过索引knn找到对应的标签

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # TODO: 统计k个最近邻的标签中出现次数最多的标签
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            y_pred[i]=np.argmax(np.bincount(closest_y.astype(int)))
            # np.bincount统计每个标签出现的次数
            # closest_y.astype(int)将标签转换为整数类型
            # np.argmax找到出现次数最多的标签的索引

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
