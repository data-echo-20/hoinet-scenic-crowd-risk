# -*- coding: utf-8 -*-
"""把多幅真实人群图像上的观测模型实例纵向拼成一张论文用图。

每幅图像占一行：左栏是 18 个格点上的真实人数（状态量），右栏是 7 个基站
扇区上由式(12) 得到的设备数（观测量）。行首一条说明只陈述本例的数据本身
（分辨率、标注人数、占用格数、信令设备数、锚点计数、反推结果），不涉及
图像的来源信息。

用法：
    python scripts/real_case_figure_stack.py \
        --item ../case_a:img_0174c --item ../case_b:scenic_s6_153 \
        --item ../case_c:scenic_s7_29 --out ../figs/fig_realcase.png
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

# 画布按**印刷尺寸**定：本图在论文里排在 \textwidth 上（sn-jnl 单栏为
# 372 pt = 5.17 in），故设计宽度就取 5.17 in，字号即为印出来的点数，不再
# 二次缩放。若仍按 11 in 作画再缩到 \textwidth，所有标注都会缩掉 53%，
# 9 pt 的说明文字印出来只剩 4.3 pt，无法阅读。
W = 5.17
L_MARGIN, R_MARGIN = 0.008, 0.992
GAP = 0.010
TITLE_BAND = 0.36
HEAD_BAND = 0.34
ROW_GAP = 0.05
BOTTOM_PAD = 0.04
NOTE_SIZE = 7.2
HEAD_SIZE = 6.8


def _load(path: str) -> np.ndarray:
    im = cv2.imread(path)
    if im is None:
        raise SystemExit('读不到 %s' % path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _numbers(values) -> str:
    return ', '.join('%d' % int(v) for v in values)


def head_line(idx: int, rec: dict, zh: bool = False) -> str:
    """行首说明：只写本例的数据，不写图像出处。"""
    cells = np.array(rec['cell_counts'])
    occ = int((cells > 0).sum())
    lo, hi = int(cells.min()), int(cells.max())
    if zh:
        return (
            '图像 %d   %d$\\times$%d 像素，标注 %d 人，'
            '占用 %d/18 个格，格点人数 %d 至 %d\n'
            '信令 $y^{\\mathrm{sig}}=(%s)$（共 %d 台设备）   '
            '锚点 $y^{\\mathrm{anc}}=(%s)$'
            % (idx, rec['resolution'][0], rec['resolution'][1],
               rec['real_heads'], occ, lo, hi,
               _numbers(rec['sector_counts']),
               int(sum(rec['sector_counts'])), _numbers(rec['anchor_counts'])))
    return (
        'Image %d    %d$\\times$%d px, %d heads, '
        '%d/18 cells occupied, cell headcounts %d to %d\n'
        'signaling $y^{\\mathrm{sig}}=(%s)$ (%d devices)   '
        'anchors $y^{\\mathrm{anc}}=(%s)$'
        % (idx, rec['resolution'][0], rec['resolution'][1], rec['real_heads'],
           occ, lo, hi, _numbers(rec['sector_counts']),
           int(sum(rec['sector_counts'])), _numbers(rec['anchor_counts'])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--item', action='append', required=True,
                    help='案例目录:图像名，可重复给出，按顺序成行')
    ap.add_argument('--out', default='fig_realcase.png')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--zh', action='store_true', help='出中文标注版')
    a = ap.parse_args()

    recs, lefts, rights = [], [], []
    for it in a.item:
        case, _, stem = it.rpartition(':')
        with open(os.path.join(case, stem + '.json'), encoding='utf-8') as f:
            recs.append(json.load(f))
        lefts.append(_load(os.path.join(case, stem + '_overlay.jpg')))
        rights.append(_load(os.path.join(case, stem + '_sectors.jpg')))

    panels = [(L_MARGIN + k * ((R_MARGIN - L_MARGIN - GAP) / 2.0 + GAP),
               (R_MARGIN - L_MARGIN - GAP) / 2.0) for k in range(2)]
    heights = [panels[0][1] * W / (im.shape[1] / float(im.shape[0]))
               for im in lefts]
    total_h = (TITLE_BAND + sum(HEAD_BAND + h + ROW_GAP for h in heights)
               + BOTTOM_PAD)

    if a.zh:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(W, total_h))
    cursor = total_h - TITLE_BAND
    for i, (rec, left, right, h) in enumerate(zip(recs, lefts, rights,
                                                  heights)):
        head_y = cursor
        cursor -= HEAD_BAND
        for (x, w), img in zip(panels, (left, right)):
            ax = fig.add_axes([x, (cursor - h) / total_h, w, h / total_h])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor('0.35')
        cursor -= h
        fig.text(L_MARGIN + 0.004, head_y / total_h, head_line(i + 1, rec, a.zh),
                 ha='left', va='top', fontsize=HEAD_SIZE, color='0.15',
                 linespacing=1.60)
        cursor -= ROW_GAP

    # 题头分两行：单行写全 "true headcount per cell n_k (18 states)" 会宽过
    # 半个画布，两个面板的题头在中缝处撞在一起。
    if a.zh:
        t0 = '(a) 每格真实人数\n$n_k$，共 18 个状态量'
        t1 = '(b) 每扇区设备数\n$y^{\\mathrm{sig}}_l$，共 7 个观测量'
    else:
        t0 = '(a) true headcount per cell\n$n_k$ over the 18 states'
        t1 = ('(b) device count per sector\n'
              '$y^{\\mathrm{sig}}_l$, 7 observations')
    for (x, w), t in zip(panels, (t0, t1)):
        fig.text(x + w / 2.0, (total_h - 0.04) / total_h, t, ha='center',
                 va='top', fontsize=NOTE_SIZE + 1.3, linespacing=1.15)

    fig.savefig(a.out, dpi=a.dpi)
    print('wrote %s  (%.1f x %.1f in)' % (a.out, W, total_h))
    for i, rec in enumerate(recs, 1):
        print('Image %d: %s' % (i, rec['image']))


if __name__ == '__main__':
    main()
