# -*- coding: utf-8 -*-
"""把 UCF-QNRF 真实图像的观测模型实例拼成论文用图。

左栏：(a) 18 个格点上的真实人数（状态量）
右栏：(b) 7 个基站扇区上由式(12) 得到的设备数（观测量）

两栏用的是同一张真实照片，只是叠加的划分不同，因此「18 个状态被压成 7 个
观测量」这件事在图上直接可见。

用法：
    python scripts/real_case_figure.py --case ucf_case --img img_0001 --out ../figs/fig_realcase.png
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

LABEL_SIZE = 13
NOTE_SIZE = 8.5


def _load(path: str) -> np.ndarray:
    im = cv2.imread(path)
    if im is None:
        raise SystemExit('读不到 %s' % path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='ucf_case')
    ap.add_argument('--img', default='img_0001')
    ap.add_argument('--out', default='fig_realcase.png')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--fit', action='store_true',
                    help='按图像宽高比自动定画布高度，避免宽幅图上下留白')
    ap.add_argument('--zh', action='store_true',
                    help='出中文标注版，供中文稿使用')
    ap.add_argument('--split', default='test',
                    help='图像所属划分，用于图注（test / train）')
    a = ap.parse_args()

    with open(os.path.join(a.case, a.img + '.json'), encoding='utf-8') as f:
        rec = json.load(f)
    cells = np.array(rec['cell_counts'])
    sectors = np.array(rec['sector_counts'])
    anchors = np.array(rec['anchor_counts'])

    left = _load(os.path.join(a.case, a.img + '_overlay.jpg'))
    right = _load(os.path.join(a.case, a.img + '_sectors.jpg'))

    fig_h = 4.6
    if a.fit:
        # 两块画板的可用宽度与高度占比，反解出让图像恰好铺满的高度
        panel_w = 11.0 * (0.994 - 0.006) / 2.0 - 0.1
        aspect = left.shape[1] / float(left.shape[0])
        fig_h = panel_w / aspect / (0.93 - 0.135)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, fig_h))
    # 面板角标并入题头而不是压在画面上：扇区标号本来就挤在边角，
    # 画面内的 (a)/(b) 会把它盖住。
    for ax, img in ((axes[0], left), (axes[1], right)):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor('0.35')

    if a.zh:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        t0 = '每格真实人数  $n_k$   （18 个状态量）'
        t1 = '每扇区设备数  $y^{\\mathrm{sig}}_l$   （7 个观测量）'
        caption = ('UCF-QNRF 测试图像 %s，%d$\\times$%d 像素，标注 %d 人。\n'
                   '面板 (a) 的计数来自公开发布的标注，面板 (b) 的数值由'
                   '式 (9) 在 $\\pi_k\\sim U(0.36,0.52)$ 下算得。'
                   % (rec['image'], rec['resolution'][0], rec['resolution'][1],
                      rec['real_heads']))
    else:
        t0 = 'true headcount per cell  $n_k$   (18 states)'
        t1 = ('device count per sector  $y^{\\mathrm{sig}}_l$   '
              '(7 observations)')
        caption = ('UCF-QNRF %s image %s, %d$\\times$%d px, %d annotated '
                   'heads.\nThe counts in (a) are the published annotation; '
                   'the numbers in (b) follow from Eq. (9) with '
                   '$\\pi_k\\sim U(0.36,0.52)$.'
                   % (a.split, rec['image'], rec['resolution'][0],
                      rec['resolution'][1], rec['real_heads']))
    axes[0].set_title('(a) ' + t0, fontsize=NOTE_SIZE + 1.5, pad=6)
    axes[1].set_title('(b) ' + t1, fontsize=NOTE_SIZE + 1.5, pad=6)
    fig.text(0.5, 0.012, caption, ha='center', va='bottom',
             fontsize=NOTE_SIZE, color='0.25', linespacing=1.5)

    fig.subplots_adjust(left=0.006, right=0.994, top=0.93, bottom=0.135,
                        wspace=0.02)
    fig.savefig(a.out, dpi=a.dpi)
    print('wrote %s' % a.out)
    print('cells   ', cells.ravel().tolist())
    print('sectors ', sectors.tolist(), 'sum', int(sectors.sum()))
    print('anchors ', anchors.tolist(), 'of cells', rec['anchor_cells'])


if __name__ == '__main__':
    main()
