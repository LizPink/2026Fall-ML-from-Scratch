import numpy as np
from ..tree_node._node import Node
from ._cart_classifier import CARTClassifier
from numpy.typing import NDArray

class RandomForestClassifier:
    """ 随机森林分类器
    Features
    --------
    1.并行训练决策树分类器，集成为强分类器
    2.训练每颗决策树前，随机选择训练样本（随机采样）；训练每颗决策树时，随机选择特征参与分裂。
    """
    def __init__(self, random_state:int=42, n_estimator:int=100, max_depth:None|int=None, min_samples_split:None|int=10, max_features:None|int=None):
        """ 初始化一颗随机森林
        Attributes
        ----------
        n_estimator:
            用于指定需要训练多少颗决策树
        max_depth:
            用于控制训练决策树的树深度
        min_samples_split:
            用于控制决策树的复杂度
        max_features:
            用于指定决策树在分裂时，选择的特征个数
        trees:
            集成的决策树列表
        """
        self.trees              = []
        self.rng                = np.random.default_rng(random_state)
        self.n_estimator        = n_estimator
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.max_features       = max_features


    def fit(self, X:NDArray, y:NDArray):
        """ 利用数据训练一颗随机森林
        Steps
        -----
        1.Bootstrap Samples
        2.Generate Random State for DecisionTrees
        3.Define and Train DecisionTrees
        4.Append to lists of trees and LOOP again
        """
        # 清空trees列表
        self.trees = []

        for _ in range(self.n_estimator):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree_seed = self.rng.integers(0, 2**32, dtype=np.int32)

            tree = CARTClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, random_state=int(tree_seed), max_features=self.max_features)
            tree.fit(X_sample, y_sample)

            self.trees.append(tree)


    def predict(self, X:NDArray):
        """ 让随机森林预测样本数据的标签
        """
        # shape:(n_estimators, predictions)
        predictions_matrix = np.array([tree.predict(X) for tree in self.trees])
        # Combime the Votes
        predictions = np.array([self._majority_vote(predictions_matrix[:, i]) for i in range(predictions_matrix.shape[1])])

        return predictions


    def _bootstrap_sample(self, X:NDArray, y:NDArray) -> tuple[NDArray, NDArray]:
        """ 随机抽取样本
        Commemts
        --------
            实现的是有放回地抽取样本
        """
        n_samples = X.shape[0]
        indices = self.rng.choice(n_samples, size = n_samples, replace = True)

        return X[indices], y[indices]


    def _majority_vote(self, votes):
        """ 返回同一个样本预测中，投票最多的标签，相同数量情况下返回数值相对较小的标签
        """
        labels, counts = np.unique(votes, return_counts=True)
        label = labels[counts.argmax()]
        return label