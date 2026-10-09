import numpy as np
from ..decision_tree._cart_regressor import CARTRegressor
from numpy.typing import NDArray

class GBDTRegressor:
    """ 梯度提升树
    Features
    --------
    1.串行训练多颗弱学习器，集成为强学习器
    2.采用L2（MSE）损失作为模型的损失函数
    """

    def __init__(self, n_estimators:int=100, learning_rate:float=0.1, max_depth:int=2, min_samples_split=2) -> None:
        """ 初始化一个GBDT学习器对象
        Attributes
        ----------
        n_estimators:
            指定弱学习器的数量
        learning_rate:
            函数梯度迭代的学习率
        max_depth:
            控制弱学习器的最大深度
        """
        self.n_estimators       = n_estimators
        self.learning_rate      = learning_rate
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        
        self.init_prediction:None|float = None
        self.trees:list[CARTRegressor]  = []


    def fit(self, X:NDArray, y:NDArray) -> None:
        # 每次训练均重新开始
        self.trees = []
        self.init_prediction = float(y.mean().astype(np.float32))

        # 开始训练
        y_pred = np.full(y.shape, self.init_prediction, dtype=np.float32)
        for _ in range(self.n_estimators):
            ## 计算函数梯度
            residual = y - y_pred

            ## 新建树拟合这部分残差
            tree = CARTRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X=X, y=residual)

            ## 更新预测函数
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)


    def predict(self, X:NDArray) -> NDArray:
        """ 利用样本特征预测标签
        Implementation
        --------------
        1.进行基准预测
        2.遍历弱学习器，逐步更新预测值
        """
        if self.init_prediction is None or self.trees == []:
            raise RuntimeError("请先训练模型，然后再预测模型.")

        prediction = np.full(X.shape[0], self.init_prediction, dtype=np.float32)
        # 逐个利用弱学习器更新预测值
        for tree in self.trees:
            tree.predict(X)
            prediction += self.learning_rate * tree.predict(X)

        return prediction
        