# -*- coding: utf-8 -*-
"""Scenic-MM 基准数据集的张量结构与读取入口，对应论文 §4.1。

Scenic-MM 随仓库以 ``data/scenic_mm.pt`` 分发，本模块定义它的张量结构
并把它读成可用的对象。数据集的规模与论文 §4.1 一致：

    18 个功能子区域，12 路摄像头覆盖其中 12 个，另 6 个为缺口区
    7 个基站扇区，15 min 聚合
    序列长度 96（15 min 步长，对应 24 h），连续 60 天，共 5760 个时间步
    视频每 60 s 一帧，12 路 x 60 天 x 1440 帧/天
    外部特征 8 维
    锚点 3 个出入口闸机 + 2 个人工抽样点位，每日两次
    风险标签四类，低/中/高/极高
    训练 42 天 / 验证 9 天 / 测试 9 天，区间不重叠

读取方式：

    from src.data import load_scenario
    sc = load_scenario('data/scenic_mm.pt')

观测模型本身（覆盖、混合、聚合、锚点、图结构）在 video.py、signaling.py、
anchors.py、graph.py 中实现，与论文 §3.3、§3.4 的观测方程一一对应。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import torch

from . import signaling as sig

# 论文 §4.1 列出的 18 个功能子区域，后 6 个为缺口区
REGION_NAMES: List[str] = [
    '入口广场', '主游览道北段', '主游览道南段', '观景平台', '中心服务区',
    '文创街区', '东侧停车场', '西侧停车场', '生态湿地区', '亲水栈道',
    '山门集散区', '游客中心', '北门集散区', '索道上下站', '演艺广场',
    '后山步道', '临时展区', '应急疏散通道',
]
GAP_REGION_NAMES: List[str] = [
    '后山步道', '东侧停车场', '生态湿地区', '文创街区', '北门集散区', '临时展区',
]


@dataclass
class ScenarioConfig:
    """场景规模，取值取自论文 §4.1 的 Scenic-MM 描述。"""

    n_regions: int = 18
    n_cameras: int = 12
    n_sectors: int = 7
    n_anchors_gate: int = 3
    n_anchors_manual: int = 2
    days: int = 60
    steps_per_day: int = 96                 # 15 min 步长 -> 96 步/天
    n_ext: int = 8
    n_classes: int = 4


@dataclass
class Scenario:
    """基准数据集里的全部张量与结构矩阵。"""

    cfg: ScenarioConfig
    n_true: torch.Tensor            # (T_total, N) 真实人数
    rho_true: torch.Tensor          # (T_total, N) 真实密度
    pi_true: torch.Tensor           # (T_total, N) 真实渗透率
    risk: torch.Tensor              # (T_total, N) 风险等级 1..4
    ext: torch.Tensor               # (T_total, E) 外部变量
    y_vis: torch.Tensor             # (T_total, M) 视觉观测
    m_vis: torch.Tensor             # (T_total, M) 有效观测权重
    o_vis: torch.Tensor             # (T_total, M) 遮挡/质量指标
    y_sig: torch.Tensor             # (T_total, J) 信令观测（按基本步窗口）
    y_anc: torch.Tensor             # (T_total, A) 锚点人数观测
    anc_mask: torch.Tensor          # (T_total, A) 锚点可用性掩码
    coverage: torch.Tensor          # (M, N) 覆盖矩阵 H
    mixing: torch.Tensor            # (L, N) 空间混合矩阵 B
    time_agg: torch.Tensor          # (J, T) 时间聚合矩阵 W
    anchor_sel: torch.Tensor        # (A, N) 锚点选择矩阵 P
    areas: torch.Tensor             # (N,) 有效面积
    adjacency: torch.Tensor         # (N, N) 功能区邻接矩阵
    laplacian: torch.Tensor         # (N, N) 已缩放的 L~
    tau: torch.Tensor               # (N, 3) 逐区域阈值
    tau_capacity: torch.Tensor      # (N, 3) 由承载量核定的阈值（部署口径）
    covered_mask: torch.Tensor      # (N,) 覆盖区掩码
    gap_mask: torch.Tensor          # (N,) 缺口区掩码
    region_names: List[str] = field(default_factory=list)


def load_scenario(path: str, window_minutes: int = 15,
                  step_minutes: int = 15) -> Scenario:
    """读取 Scenic-MM 数据集文件。

    Args:
        path: data/scenic_mm.pt 的路径。
        window_minutes: 信令聚合窗口；文件里记有该值，显式传入可覆盖。
        step_minutes: 基本步长。

    Returns:
        Scenario 对象。时间聚合矩阵 W 由 signaling.time_aggregation_matrix
        按窗口长度重建——它只取决于步长与窗口，与数据无关，故文件中不重复
        存储这份 (T_total, T_total) 的结构矩阵。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            '找不到数据集 %s；Scenic-MM 随仓库分发，见 README 的说明' % path)
    blob = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ScenarioConfig(**blob['config'])
    state = dict(blob['state_dict'])
    t_total = cfg.days * cfg.steps_per_day
    state['time_agg'] = sig.time_aggregation_matrix(
        t_total, step_minutes=blob.get('step_minutes', step_minutes),
        window_minutes=blob.get('window_minutes', window_minutes))
    return Scenario(cfg=cfg, **state)
