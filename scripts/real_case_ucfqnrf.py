# -*- coding: utf-8 -*-
"""在真实人群图像上实例化论文 §3.3 的异质观测模型。

与 Scenic-MM 不同的是，这个案例的**状态量是真的**：人数与空间分布直接取自
公开数据集的点标注（每张图给出每个头部的像素坐标），不是模拟出来的。论文
§4.1 的 fig:realcase 三行分别是一帧 UCF-QNRF 与两帧 CrowdUIT，表 1 另有四帧
UCF-QNRF；两类标注的格式相同，都是单个 (N,2) 的 float64 数组 annPoints，故
本脚本对二者通用，文件名不带数据集前缀只是沿用最初的命名。

信令通道与锚点通道按论文自己的观测算子由该真实状态生成：

    式(12)   y^sig_l = sum_k B_{l,k} * pi_k * n_k
    式(14)  y^anc_a = sum_k P_{a,k} * n_k

因此本脚本模拟的只有观测过程，状态本身可回溯到公开数据集的标注文件。

分区口径
--------
每张图按 3x6 均分为 18 个像元格，与 Scenic-MM 的状态维数 N=18 对齐。该分区是
**像平面上的均匀划分**，不是功能子区划分：透视投影会把远处的人群压缩到画面的
上缘，因此实测的格点人数沿竖直方向极不均匀（UCF-QNRF 的 img_0217 有 67% 的
头部落在最上一行）。这正是 Scenic-MM 采用场地平面分区而不是图像分区的原因，
本案例把这个差别如实报出来。

信令分区
--------
7 个基站扇区取为以画面底部中心为顶点的一组角形楔区，覆盖整个画面。B 按面积
份额构造：B_{l,k} = |格 k ∩ 楔 l| / |格 k|，列和恒为 1，与论文 B 的列随机性质
一致。渗透率 pi_k 在论文 §4.1 给出的 0.36—0.52 区间内按区域抽取。

数据获取
--------
UCF-QNRF 需从官方渠道（https://www.crowd-counting.com/ 的下载页）申请获取，
仓库不分发图像与标注。取到压缩包后只需其中任意若干张测试图像：

    UCF-QNRF_ECCV18.zip
    ├── Train/  img_0001.jpg … img_1201.jpg 及同名 _ann.mat
    └── Test/   img_0001.jpg … img_0334.jpg 及同名 _ann.mat

把选中的 .jpg 与对应的 _ann.mat 放到同一目录（下例中的 img/）即可运行。标注
文件内是单个 (N,2) 的 float64 数组 annPoints，即每个头部的像素坐标。压缩包
约 4.5 GB，若只想取几张图，可先读中央目录再按 range 请求单条解压，不必整包下载。

CrowdUIT（论文引用的另一来源，见 bib 中的 refcrowduit）已公开在 Kaggle：
https://www.kaggle.com/datasets/khitthanhnguynphan/crowduit 。它的 scenic_*
图像同样并列给出 .jpg 与 _ann.mat，annPoints 格式与 UCF-QNRF 一致，直接按
同一条命令即可处理，无需改动脚本。

用法：
    python scripts/real_case_ucfqnrf.py --img img/img_0001.jpg --ann img/img_0001_ann.mat \
        --out case/ --seed 20260913
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from scipy.io import loadmat
from scipy.optimize import nnls

SEED = 20260913
GRID_ROWS, GRID_COLS = 3, 6
N_SECTORS = 7
PI_LO, PI_HI = 0.36, 0.52          # 论文 §4.1 的渗透率区间
# 锚点：3 个闸机点位落在最下一行的左/中/右三格，2 个人工抽样点位落在中间行
GATE_CELLS = (12, 15, 17)
MANUAL_CELLS = (7, 10)


# ------------------------------------------------------------------ 状态
def cell_counts(points: np.ndarray, width: int, height: int
                ) -> np.ndarray:
    """把真实头部坐标落到 3x6 网格上，返回 (3, 6) 的真实格点人数。"""
    gx = np.clip((points[:, 0] / width * GRID_COLS).astype(int),
                 0, GRID_COLS - 1)
    gy = np.clip((points[:, 1] / height * GRID_ROWS).astype(int),
                 0, GRID_ROWS - 1)
    return np.bincount(gy * GRID_COLS + gx,
                       minlength=GRID_ROWS * GRID_COLS
                       ).reshape(GRID_ROWS, GRID_COLS)


# ------------------------------------------------------------------ 混合矩阵
def sector_mixing(width: int, height: int, n_sectors: int = N_SECTORS
                  ) -> np.ndarray:
    """构造 (L, N) 的空间混合矩阵 B，列为各格落在 7 个扇区上的面积份额。

    扇区是以画面底部中心为顶点的角形楔区，等角分割 180 度张角；顶点略高于
    底边，使最近的几格也落在扇区内。逐像素判定归属后按格统计面积份额。
    """
    apex = np.array([width / 2.0, height * 1.05])
    yy, xx = np.mgrid[0:height, 0:width]
    dx = xx - apex[0]
    dy = apex[1] - yy                       # 向上为正
    ang = np.arctan2(dx, np.maximum(dy, 1e-6))       # -pi/2 .. +pi/2
    edges = np.linspace(-np.pi / 2, np.pi / 2, n_sectors + 1)
    sector = np.clip(np.digitize(ang, edges[1:-1]), 0, n_sectors - 1)

    cell_x = np.clip((xx / width * GRID_COLS).astype(int), 0, GRID_COLS - 1)
    cell_y = np.clip((yy / height * GRID_ROWS).astype(int), 0, GRID_ROWS - 1)
    cell = cell_y * GRID_COLS + cell_x

    b = np.zeros((n_sectors, GRID_ROWS * GRID_COLS))
    for l in range(n_sectors):
        for k in range(GRID_ROWS * GRID_COLS):
            area = float(((cell == k) & (sector == l)).sum())
            total = float((cell == k).sum())
            b[l, k] = area / max(total, 1.0)
    # 列归一，吸收浮点误差，保证与论文 B 的列随机性质严格一致
    return b / np.maximum(b.sum(axis=0, keepdims=True), 1e-12)


# ------------------------------------------------------------------ 观测
def simulate(signal_n: np.ndarray, mixing: np.ndarray, seed: int = SEED):
    """按式(12) 与式(14) 由真实格点人数生成信令与锚点观测。

    返回 (sector_counts, penetration, anchor_counts)。
    """
    rng = np.random.default_rng(seed)
    n_cells = signal_n.size
    pi = rng.uniform(PI_LO, PI_HI, size=n_cells)
    devices = pi * signal_n
    sector = mixing @ devices
    gate = np.array([signal_n[k] for k in GATE_CELLS], dtype=float)
    manual = np.array([signal_n[k] for k in MANUAL_CELLS], dtype=float)
    return np.round(sector), pi, np.concatenate([gate, manual])


def signaling_only(state_n: np.ndarray, mixing: np.ndarray,
                   sector_counts: np.ndarray, pi_nominal: float = 0.45):
    """只用信令观测按非负最小二乘反推状态，用于量化信令通道的欠定性。

    渗透率在测试期不可知，与人数构成尺度耦合（信令只约束二者之积，不约束各自
    的绝对量级）；这里取论文 §4.1 报告的均值 0.45 作为标称值。18 个未知量、
    7 个方程，方程数远少于未知量，故残差
    可以压到零，但解不唯一——NNLS 给出的只是其中一族极端稀疏的解。

    返回 (估计状态, 残差, 逐格相对误差, 零格数)。
    """
    a = mixing * pi_nominal
    est, res = nnls(a, sector_counts)
    rel = np.abs(est - state_n) / np.maximum(state_n, 1.0)
    return est, float(res), rel, int((est < 1.0).sum())


# ------------------------------------------------------------------ 可视化
def overlay(image: np.ndarray, counts: np.ndarray, points: np.ndarray,
            path: str) -> None:
    """把格点分区、格点人数与真实头部位置画在原图上。"""
    h, w = image.shape[:2]
    out = image.copy()
    flat = counts.ravel()
    vmax = max(float(flat.max()), 1.0)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            k = r * GRID_COLS + c
            x0, x1 = int(c * w / GRID_COLS), int((c + 1) * w / GRID_COLS)
            y0, y1 = int(r * h / GRID_ROWS), int((r + 1) * h / GRID_ROWS)
            t = flat[k] / vmax
            colour = np.array([40 + 215 * t, 210 - 190 * t, 60 - 40 * t])
            # 叠色强度随人数走：人少的格几乎保留原照片，人多的格才明显染色，
            # 避免整幅图被色彩层压掉，场景本身仍看得清。
            alpha = 0.06 + 0.24 * t
            roi = out[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = ((1.0 - alpha) * roi
                                 + alpha * colour.astype(np.uint8)).astype(
                                     np.uint8)
            cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 2)
            # 字号按「印刷后的点数」定，而不是按源图像素定：本图在论文里排在
            # \textwidth 上，面板印宽固定，故字体像素高与源图宽成正比时，
            # 印出来的点数反而是常数。除以 620 使格点人数印出来约 7.4 pt、
            # 扇区标号约 6.8 pt，落在期刊要求的可读区间的下沿之上。
            scale = w / 620.0
            cv2.putText(out, str(int(flat[k])), (x0 + int(10 * scale),
                        y0 + int(38 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.15 * scale,
                        (0, 0, 0), int(5 * scale), cv2.LINE_AA)
            cv2.putText(out, str(int(flat[k])), (x0 + int(10 * scale),
                        y0 + int(38 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.15 * scale,
                        (255, 255, 255), int(2 * scale), cv2.LINE_AA)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 90])


SECTOR_COLOURS = [(214, 39, 40), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (23, 190, 207), (31, 119, 180),
                  (148, 103, 189)]


def sector_edges(width: int, height: int, n_sectors: int = N_SECTORS):
    """扇区边界在地面上的两个端点，用于画楔区轮廓。"""
    apex = np.array([width / 2.0, height * 1.05])
    edges = np.linspace(-np.pi / 2, np.pi / 2, n_sectors + 1)
    pts = []
    for e in edges:
        far = apex + np.array([np.sin(e), -np.cos(e)]) * (height * 3.0)
        pts.append((apex, far))
    return pts


def overlay_sectors(image: np.ndarray, sector_counts: np.ndarray, path: str
                    ) -> None:
    """把 7 个基站扇区画在原图上，并标出该扇区由式(12) 得到的设备数。"""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    apex = np.array([w / 2.0, h * 1.05])
    ang = np.arctan2(xx - apex[0], np.maximum(apex[1] - yy, 1e-6))
    edges = np.linspace(-np.pi / 2, np.pi / 2, N_SECTORS + 1)
    sector = np.clip(np.digitize(ang, edges[1:-1]), 0, N_SECTORS - 1)

    out = image.copy()
    for l in range(N_SECTORS):
        m = sector == l
        col = np.array(SECTOR_COLOURS[l], dtype=np.uint8)
        out[m] = (0.78 * out[m] + 0.22 * col).astype(np.uint8)
    for apex_pt, far in sector_edges(w, h):
        cv2.line(out, tuple(np.round(apex_pt).astype(int)),
                 tuple(np.round(far).astype(int)), (255, 255, 255),
                 max(2, int(w / 400)), cv2.LINE_AA)
    # 标号落在各自楔区可见部分的重心上；靠下的几个楔区又窄又挤，重心会撞到
    # 一起，所以再走一遍避让：与已放置的标号重叠就整体下移，保证 7 个数值都读得到。
    # 除数与 overlay() 一致，见那里的说明。
    scale = w / 620.0
    placed = []
    for l in range(N_SECTORS):
        ys, xs = np.nonzero(sector == l)
        if xs.size == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        txt = 'S%d  %d' % (l + 1, int(sector_counts[l]))
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                      1.05 * scale, int(3 * scale))
        step = th + int(14 * scale)
        while any(abs(cx - px) < (tw + pw) // 2 + int(8 * scale)
                  and abs(cy - py) < (th + ph) // 2 + int(8 * scale)
                  for px, py, pw, ph in placed):
            if cy + step + int(8 * scale) > h:
                cy -= step
                break
            cy += step
        placed.append((cx, cy, tw, th))
        cv2.rectangle(out, (cx - tw // 2 - int(8 * scale),
                            cy - th - int(8 * scale)),
                      (cx + tw // 2 + int(8 * scale),
                       cy + int(8 * scale)), (0, 0, 0), -1)
        cv2.putText(out, txt, (cx - tw // 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05 * scale,
                    (255, 255, 255), int(3 * scale), cv2.LINE_AA)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    ap = argparse.ArgumentParser(description='UCF-QNRF 真实图像上的观测模型实例')
    ap.add_argument('--img', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--out', default='case')
    ap.add_argument('--seed', type=int, default=SEED)
    a = ap.parse_args()

    image = cv2.imread(a.img)
    if image is None:
        raise SystemExit('读不到图像 %s' % a.img)
    h, w = image.shape[:2]
    points = loadmat(a.ann)['annPoints']

    counts = cell_counts(points, w, h)
    mixing = sector_mixing(w, h)
    sector, pi, anchor = simulate(counts.ravel(), mixing, a.seed)

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.img))[0]
    overlay(image, counts, points, os.path.join(a.out, stem + '_overlay.jpg'))
    overlay_sectors(image, sector,
                    os.path.join(a.out, stem + '_sectors.jpg'))

    # B 的条件数：把「信令能不能定住状态」这件事量化
    cond = float(np.linalg.cond(mixing))
    est, res, rel, n_zero = signaling_only(counts.ravel(), mixing, sector)
    record = {
        'image': os.path.basename(a.img),
        'resolution': [int(w), int(h)],
        'real_heads': int(points.shape[0]),
        'cell_counts': counts.tolist(),
        'penetration': [round(float(v), 4) for v in pi],
        'sector_counts': [int(v) for v in sector],
        'anchor_counts': [int(v) for v in anchor],
        'anchor_cells': list(GATE_CELLS) + list(MANUAL_CELLS),
        'mixing_condition_number': round(cond, 3),
        'top_row_share': round(float(counts[0].sum() / counts.sum()), 4),
        'signaling_only': {
            'state_estimate': [round(float(v), 2) for v in est],
            'residual': round(res, 4),
            'median_relative_error': round(float(np.median(rel)), 4),
            'max_relative_error': round(float(rel.max()), 4),
            'zero_cells': n_zero,
            'total_bias': round(float(est.sum() / counts.sum() - 1.0), 4),
        },
    }
    with open(os.path.join(a.out, stem + '.json'), 'w',
              encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
