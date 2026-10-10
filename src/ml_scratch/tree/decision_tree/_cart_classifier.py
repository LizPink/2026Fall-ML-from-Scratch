import numpy as np
from numpy.typing import NDArray
from ..tree_node._node import Node


class CARTClassifier:
    """ CART Decision Tree Classifier
    1.采用Gini不纯度决定分裂策略
    2.采用0-1损失作为全局损失函数——叶子节点采用多数类标签作为预测标签
    """
    def __init__(self, max_depth:None|int=2, min_samples_split:None|int=10, random_state:None|int=None, max_features:None|int=None):
        """ 初始化一个CART算法的决策分类树
        Attributes
        ----------
        random_state:
            为了配合RandomForest的随机性，用于控制每棵树的随机行为，与max_features等属性有关。如果为None表示不启用随机性。
        max_features:
            为了配合RandomForest的随机性，每次节点分裂时，随机选择的特征数，如果为None就是表示全量特征参与。
        """
        self.root               = None
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.rng                = np.random.default_rng(random_state) if random_state is not None else None
        self.max_features       = max_features


    def fit(self, X:NDArray, y:NDArray) -> None:
        self.root = self._build_tree(X, y, depth=0)


    def predict(self, X:NDArray) -> NDArray:
        """ 接收预测样本的特征，返回预测样本的标签
        Implementation
        --------------
        1.循环遍历每条样本
        2.逐条样本进行预测
        3.预测结果汇总返回
        """
        if self.root is None:
            raise RuntimeError("请先进行训练，然后再进行预测")
        
        predictions = []
        for idx in range(X.shape[0]):
            label = self._predict_one(X[idx, :])
            predictions.append(label)

        return np.array(predictions)


    def _build_tree(self, X:NDArray, y:NDArray, depth:int) -> Node:
        """ 构建决策分类树
        Steps
        -----
        1.判断分裂是否终止
        2.寻找最佳分裂特征和阈值
        3.递归构建左右子树
        """
        if self._should_stop(X, y, depth):
            return self._make_leaf(y)

        feature, threshold = self._best_split(X, y)
        left_mask = X[:,feature] <= threshold
        right_mask = ~left_mask

        ## 递归的"递"
        left_child = self._build_tree(X=X[left_mask], y=y[left_mask], depth=depth+1)
        right_child = self._build_tree(X=X[right_mask], y=y[right_mask], depth=depth+1)

        ## 递归的"归"
        return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)
        
        
    def _should_stop(self, X:NDArray, y:NDArray, depth:int) -> bool:
        """ 判断一个节点是否需要分裂
        """
        # 分裂是否达到最大深度
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        # 样本是否足够多
        if self.min_samples_split is not None and y.shape[0] < self.min_samples_split:
            return True
        # 样本标签是否足够纯净
        if np.unique(y).size == 1:
            return True
        # 样本特征是否完全相同
        if np.unique(X, axis=0).shape[0] == 1:
            return True

        return False


    def _make_leaf(self, y:NDArray) -> Node:
        """ 将该节点定义为一个叶节点
        """
        labels, counts = np.unique(y, return_counts=True, sorted=True)
        majority_class = int(labels[counts.argmax()])

        return Node(value=majority_class)


    def _sample_features(self, n_features:int) -> NDArray:
        """ 返回随机采样的候选特征列
        """
        assert self.rng is not None

        if self.max_features is None:
            return np.arange(n_features)
        
        sampled_features = self.rng.choice(range(n_features), n_features, replace=False)
        return sampled_features


    def _best_split(self, X:NDArray, y:NDArray) -> tuple[int, float]:
        """ 依次搜索特征和阈值，进行评分，记录并更新评分最高的（特征，阈值）组合
        """
        best_score = np.inf
        best_feature = None
        best_threshold = None

        for feature in self._sample_features(X.shape[1]):
            values = np.unique(X[:,feature], sorted=True)
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:,feature] <= threshold
                right_mask = ~left_mask
                y_left = y[left_mask]
                y_right = y[right_mask]
                split_score = self._split_score(y_left=y_left, y_right=y_right)

                if split_score < best_score:
                    best_score = split_score
                    best_feature = feature
                    best_threshold = threshold

        assert best_feature is not None and best_threshold is not None
        return int(best_feature), float(best_threshold)


    def _split_score(self, y_left:NDArray, y_right:NDArray) -> float:
        """ 给定特征和阈值，以及划分的样本，返回样本划分的评分
        Implementation
        --------------
        在CART树中，分裂的评分采用的是Gini系数进行计算.
        """
        n_left, n_right, n_total = len(y_left), len(y_right), len(y_left)+len(y_right)
        gini_left, gini_right = self._gini(y_left), self._gini(y_right)

        left_weight, right_weight = n_left/n_total, n_right/n_total
        gini_weighted = left_weight*gini_left + right_weight*gini_right

        return gini_weighted


    def _gini(self, y:NDArray) -> float:
        """ 计算一组标签数据的Gini不纯度
        """
        values, counts = np.unique(y, return_counts=True, sorted=True)
        frequency_squared = (counts / counts.sum()) ** 2
        gini = 1 - frequency_squared.sum()

        return gini


    def _predict_one(self, x:NDArray) -> int:
        """ 预测单条样本数据
        Implementation
        --------------
        1.循环直至走到叶子节点
        2.返回叶子节点样本中的value
        """
        node = self.root
        assert node is not None

        while(node.value is None):
            if x[node.feature] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right

        assert isinstance(node.value, int)
        return node.value