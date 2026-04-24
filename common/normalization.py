import numpy as np


class RunningMeanStd:
    """在线计算均值和标准差（Welford 算法）。"""

    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class Normalization:
    """
    对输入向量做零均值单位方差归一化。

    用法：
        norm = Normalization(shape=30)
        x_normed = norm(x)            # 训练时：更新统计量并归一化
        x_normed = norm(x, update=False)  # 评估时：仅归一化，不更新统计量
    """

    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x

class RewardScaling:
    """
    基于折扣回报的奖励缩放（只除以标准差，不减均值）。

    每步调用：r_scaled = r / std(R)，其中 R = γ·R + r 是运行折扣累积回报。
    episode 结束时调用 reset() 清空 R。
    """

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=1)
        self.R = np.zeros(1)

    def __call__(self, r: float) -> float:
        self.R = self.gamma * self.R + r
        self.running_ms.update(self.R)
        return float(r / (self.running_ms.std + 1e-8))

    def reset(self):
        self.R = np.zeros(1)


