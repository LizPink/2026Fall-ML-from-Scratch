from dataclasses import dataclass

@dataclass
class Node:
    """ 树类模型的基础节点
    Attributes
    ----------
    value
        如果有则代表这是一个叶子节点，则value为预测值
    """
    feature:    None|int        = None
    threshold:  None|float      = None
    left:       None|Node       = None
    right:      None|Node       = None
    value:      None|int|float  = None