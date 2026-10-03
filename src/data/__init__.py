# -*- coding: utf-8 -*-
"""数据侧：图结构、信令、视频、锚点、标签与 Scenic-MM 基准。

模块与论文小节的对应：
    graph.py       §3.4    功能区邻接与图拉普拉斯
    signaling.py   §3.3    式(12) 信令的时空聚合与脱敏
    video.py       §3.3    式(9)(10) 视频覆盖与异方差噪声
    anchors.py     §3.3    式(14) 锚点观测与渗透率标定
    labels.py      §3.1    式(3) 拥堵持续状态与风险阈值
    scenario.py    §4.1    Scenic-MM 基准的张量结构与读取入口
    dataset.py     §4.1    42/9/9 天划分与序列滑窗
"""

from .dataset import (ScenarioTensors, ScenicWindowDataset, SplitIndex,
                      collate, default_splits, make_loaders)
from .scenario import Scenario, ScenarioConfig, load_scenario

__all__ = ['Scenario', 'ScenarioConfig', 'load_scenario',
           'ScenicWindowDataset', 'ScenarioTensors', 'SplitIndex',
           'default_splits', 'make_loaders', 'collate']
