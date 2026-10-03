# -*- coding: utf-8 -*-
"""Scenic-MM 视频通道渲染器。

把基准的真值人数、覆盖矩阵与有效掩码渲染成每一路摄像头真正会看到的画面，
使 Scenic-MM 从「一组数值张量」变成「带真实图像的基准」。

对应论文 §3.3 的视频通道式(9)—式(11)：渲染帧给出观测算子
sum_k H[m,k]*n[t,k] 的真值侧，基准中存的 y_vis 则在其上叠加了计数噪声。

渲染口径（可逐句与论文正文核对）：
  1. 场地上的 18 个功能子区按邻接环的顺序排布在一条闭合游线上，
     子区面积取基准中的 areas，取正方形（边长由面积开方）。
  2. 每个子区按真值人数 n_true[t, k] 撒点（中心略聚集），
     摄像头 m 看到的是「离子区中心最近的 round(H[m,k]·n[k]) 个」，
     因此画面中的真实人数严格等于观测算子 sum_k H[m,k]·n[k] 的取整结果。
  3. 基准里存的 y_vis 是在该真值上叠加了计数噪声后的视频观测，
     渲染帧给出的是它的真值侧，两者之差就是观测噪声本身。
  4. 摄像头在时刻 t 无效（m_vis = 0）时输出无信号帧。
  5. 画面经过时变光照、夜间灯位、深度模糊、传感器噪声与暗角处理。

用法：
    python scripts/render_frames.py --pt data/scenic_mm.pt --probe 4
    python scripts/render_frames.py --pt data/scenic_mm.pt --out frames
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

SEED = 20260913
PPM = 9.0               # 地面栅格：像素/米
IMG_W, IMG_H = 800, 450
MARGIN = 6.0            # 地面窗口在可视区外的留边（米）

TERRAIN = {
    '入口广场':     ((176, 178, 180), 'pave'),
    '主游览道北段': ((152, 150, 146), 'path'),
    '主游览道南段': ((152, 150, 146), 'path'),
    '观景平台':     ((196, 196, 192), 'pave'),
    '中心服务区':   ((168, 168, 166), 'pave'),
    '文创街区':     ((164, 166, 176), 'pave'),
    '东侧停车场':   ((96, 98, 100), 'park'),
    '西侧停车场':   ((96, 98, 100), 'park'),
    '生态湿地区':   ((118, 150, 116), 'grass'),
    '亲水栈道':     ((122, 152, 186), 'water'),
    '山门集散区':   ((176, 178, 180), 'pave'),
    '游客中心':     ((172, 172, 170), 'pave'),
    '北门集散区':   ((176, 178, 180), 'pave'),
    '索道上下站':   ((160, 158, 156), 'pave'),
    '演艺广场':     ((182, 184, 186), 'pave'),
    '后山步道':     ((128, 142, 132), 'trail'),
    '临时展区':     ((170, 170, 168), 'pave'),
    '应急疏散通道': ((150, 148, 144), 'path'),
}
BUILDINGS = ['中心服务区', '文创街区', '游客中心', '索道上下站', '临时展区']

# 衣着配色（BGR）：深浅灰、藏青、白、红、墨绿、卡其、黑
CLOTHES = np.array([[52, 52, 56], [30, 30, 34], [128, 72, 34], [205, 208, 212],
                    [40, 44, 122], [46, 46, 150], [72, 96, 58], [150, 158, 178],
                    [96, 92, 88], [180, 176, 168], [60, 60, 64], [34, 88, 118]],
                   np.float32)


@dataclass
class Zone:
    name: str
    area: float
    center: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    length: float
    depth: float
    quad: np.ndarray


# ------------------------------------------------------------ 场地与摄像头

def build_layout(names, areas):
    """18 个子区沿一条闭合游线（椭圆）排布，顺序即邻接环的顺序。

    子区取正方形（边长由面积开方），环的周长留出余量使最大的子区也放得下。
    """
    a, b = 300.0, 260.0
    per = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    pitch = per / len(names)
    zones = []
    for i, (nm, ar) in enumerate(zip(names, areas)):
        phi = 2 * np.pi * i / len(names)
        c = np.array([a * np.cos(phi), b * np.sin(phi)])
        t = np.array([-a * np.sin(phi), b * np.cos(phi)])
        t /= np.linalg.norm(t)
        n = np.array([b * np.cos(phi), a * np.sin(phi)])
        n /= np.linalg.norm(n)
        if np.linalg.norm(c + n) < np.linalg.norm(c - n):
            n = -n
        ln = dp = float(np.clip(np.sqrt(ar), 24.0, 0.86 * pitch))
        q = np.array([c + t * ln / 2 + n * dp / 2, c - t * ln / 2 + n * dp / 2,
                      c - t * ln / 2 - n * dp / 2, c + t * ln / 2 - n * dp / 2])
        zones.append(Zone(nm, float(ar), c, t, n, ln, dp, q))
    return zones


def sample_zone(zone, n, rng):
    """在子区内撒 n 个点，中心略聚集（|u|^1.35 收缩），返回地面坐标。"""
    u = rng.random(n) * 2 - 1
    v = rng.random(n) * 2 - 1
    su = np.sign(u) * np.abs(u) ** 1.35 * (zone.length / 2)
    sv = np.sign(v) * np.abs(v) ** 1.35 * (zone.depth / 2)
    return (zone.center[0] + su * zone.tangent[0] + sv * zone.normal[0],
            zone.center[1] + su * zone.tangent[1] + sv * zone.normal[1])


def visible_pts(zone, w, rng, n):
    """子区内离子区中心最近的 round(w*n) 个点。

    可视范围因此是子区中心处一块面积占比 w 的紧凑区域，
    面积占比严格等于覆盖矩阵的 H，且投影后形状接近画幅比例。
    """
    gx, gy = sample_zone(zone, n, rng)
    k = max(0, min(n, int(round(w * n))))
    if k == 0:
        return np.zeros((0, 2))
    d = np.hypot(gx - zone.center[0], gy - zone.center[1])
    idx = np.argpartition(d, k - 1)[:k]
    return np.stack([gx[idx], gy[idx]], 1)


class Pose:
    """针孔相机：主光轴指向目标，俯角由机位高度与水平距离确定。"""

    def __init__(self, pos_xy, height, target_xy, f_img):
        d = np.asarray(target_xy, float) - np.asarray(pos_xy, float)
        dist = max(float(np.linalg.norm(d)), 1e-6)
        yaw = float(np.arctan2(d[1], d[0]))
        pitch = float(np.arctan2(height, dist))
        self.C = np.array([pos_xy[0], pos_xy[1], height], float)
        self.f_px = f_img
        self.yaw = yaw
        self.fwd = np.array([np.cos(pitch) * np.cos(yaw),
                             np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
        self.right = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        up = np.cross(self.fwd, self.right)
        self.up = up / np.linalg.norm(up)

    def project3(self, xyz):
        p = np.atleast_2d(np.asarray(xyz, float)) - self.C
        z = p @ self.fwd
        u = self.f_px * (p @ self.right) / np.maximum(z, 1e-6)
        v = -self.f_px * (p @ self.up) / np.maximum(z, 1e-6)
        return np.stack([u + IMG_W / 2, v + IMG_H / 2], 1), z

    def project(self, xy):
        """地面点（高程 0）的投影。"""
        xy = np.atleast_2d(np.asarray(xy, float))
        return self.project3(np.c_[xy, np.zeros(len(xy))])


def prepare(m, zones, H):
    """定出第 m 路摄像头的地面参考点、可视范围、渲染位姿与地面窗口。

    机位沿主辖子区的外法向退到子区外侧，退多远由「可视区半径 → 视场角」反推：
    水平视场角取 56°，使可视区恰好铺满画面，避免近端地面被透视过度放大。
    """
    targets = [(k, float(H[m, k])) for k in range(H.shape[1]) if H[m, k] > 1e-9]
    k0 = max(targets, key=lambda kv: kv[1])[0]
    z0 = zones[k0]
    ref = z0.center + z0.normal * (z0.depth / 2 + 9.0)

    rng = np.random.default_rng(SEED + 1000 + m)
    pts = np.concatenate([visible_pts(zones[k], w, rng, 8000) for k, w in targets])
    ctr = pts.mean(0)

    # 机位退到子区外侧，退多远由可视区半径与水平视场角反推；俯角约 37°
    radius = float(np.linalg.norm(pts - ctr, axis=1).max())
    back = float(np.clip(radius / np.tan(np.deg2rad(28.0)), 16.0, 200.0))
    cam_xy = ctr + (ref - ctr) / max(np.linalg.norm(ref - ctr), 1e-6) * back
    height = float(np.clip(0.75 * back, 10.0, 42.0))

    pose = Pose(cam_xy, height, ctr, 1000.0)
    uv, zz = pose.project(pts)
    ok = zz > 0.5
    if ok.sum() < 8:
        return None
    uv = uv[ok]
    pose.f_px *= min(IMG_W * 0.90 / max(float(np.ptp(uv[:, 0])), 1e-6),
                     IMG_H * 0.90 / max(float(np.ptp(uv[:, 1])), 1e-6))

    # 视线对齐的地面窗口：前向起点留 0.8 m，保证窗口四角全部位于摄像头前方
    e1 = np.array([np.cos(pose.yaw), np.sin(pose.yaw)])
    e2 = np.array([-np.sin(pose.yaw), np.cos(pose.yaw)])
    rel = pts - cam_xy
    t, s = rel @ e1, rel @ e2
    t0, t1 = max(0.8, t.min() - MARGIN), t.max() + MARGIN
    s0, s1 = s.min() - MARGIN, s.max() + MARGIN
    origin = cam_xy + e1 * t0 + e2 * s0
    # 窗口像素坐标 = (前向·e1, 横向·e2)·PPM，故宽取前向跨度、高取横向跨度
    size = (int((t1 - t0) * PPM), int((s1 - s0) * PPM))

    terr = terrain_layer(zones, origin, e1, e2, size)
    # 窗口四角的世界坐标按取整后的实际尺寸回算，供投影求单应
    w, hgt = size
    corners = np.array([origin,
                        origin + e1 * (w / PPM),
                        origin + e1 * (w / PPM) + e2 * (hgt / PPM),
                        origin + e2 * (hgt / PPM)])
    trees, build = scenery(zones, cam_xy, e1, e2, (t0, t1, s0, s1), m, ctr, k0)
    return {'m': m, 'cam_xy': cam_xy, 'targets': targets,
            'pose': pose, 'origin': origin, 'e1': e1, 'e2': e2, 'size': size,
            'terrain': terr, 'primary': k0, 'corners': corners,
            'trees': trees, 'buildings': build}


def scenery(zones, cam_xy, e1, e2, span, m, ctr, k0):
    """在可视地面窗口内落位立体物：子区之外的树木，以及带高度的建筑。

    立面与树冠在屏幕空间按真实尺寸绘制，使画面具备高度线索，
    而不是一张俯视平面图。摄像头到主辖子区之间留出通视走廊，
    避免树冠把被观测的人群挡住。
    """
    t0, t1, s0, s1 = span
    rng = np.random.default_rng(SEED + 500 + m)
    quads = [z.quad.astype(np.float32) for z in zones]
    fwd_ctr = float((ctr - cam_xy) @ e1)
    half_w = float(zones[k0].length / 2 + 5.0)

    trees = []
    n_cand = int((t1 - t0) * (s1 - s0) / 90.0)
    for tt, ss in rng.uniform([t0, s0], [t1, s1], (n_cand, 2)):
        g = cam_xy + e1 * tt + e2 * ss
        rel = g - cam_xy
        if float(rel @ e1) < fwd_ctr + 12.0 and abs(float(rel @ e2)) < half_w:
            continue                          # 通视走廊
        if any(cv2.pointPolygonTest(q, (float(g[0]), float(g[1])), False) >= 0
               for q in quads):
            continue
        trees.append((float(g[0]), float(g[1]),
                      float(rng.uniform(5.5, 10.0)), float(rng.uniform(1.8, 3.4))))

    build = []
    for z in zones:
        if z.name not in BUILDINGS:
            continue
        for _ in range(int(rng.integers(1, 3))):
            cl = z.center + z.tangent * rng.uniform(-0.22, 0.22) * z.length \
                + z.normal * rng.uniform(-0.22, 0.22) * z.depth
            ww = rng.uniform(0.26, 0.42) * z.length
            hh = rng.uniform(0.26, 0.42) * z.depth
            q = np.array([cl + z.tangent * ww / 2 + z.normal * hh / 2,
                          cl - z.tangent * ww / 2 + z.normal * hh / 2,
                          cl - z.tangent * ww / 2 - z.normal * hh / 2,
                          cl + z.tangent * ww / 2 - z.normal * hh / 2])
            build.append((q, float(rng.uniform(5.5, 9.5))))
    return trees, build


def draw_scenery(img, cam, pose):
    """把树木与建筑按高度投到画面，远景先画以形成正确的相互遮挡。"""
    items = []
    for x, y, h, rad in cam['trees']:
        uv, z = pose.project([(x, y)])
        if z[0] > 1.0:
            items.append((float(z[0]), 0, (x, y, h, rad, uv[0])))
    for q, h in cam['buildings']:
        uv, z = pose.project(q)
        if (z > 1.0).all():
            items.append((float(z.mean()), 1, (q, h, uv)))

    for _, kind, data in sorted(items, key=lambda it: -it[0]):
        if kind == 0:
            x, y, h, rad, base = data
            top, zt = pose.project3([(x, y, h)])
            if zt[0] <= 1.0:
                continue
            hw = pose.f_px * rad / zt[0]
            if hw < 1.3:
                continue
            cx, cy = int(top[0][0]), int(top[0][1])
            cv2.line(img, (int(base[0]), int(base[1])), (cx, cy),
                     (44, 50, 42), max(1, int(hw * 0.20)), cv2.LINE_AA)
            cv2.circle(img, (cx, cy), int(hw), (40, 58, 36), -1, cv2.LINE_AA)
            cv2.circle(img, (cx - int(hw * 0.22), cy - int(hw * 0.22)),
                       int(hw * 0.60), (60, 82, 50), -1, cv2.LINE_AA)
        else:
            q, h, base_uv = data
            roof_uv, zr = pose.project3(np.c_[q, np.full(4, h)])
            if (zr <= 1.0).any():
                continue
            for i in range(4):
                wall = np.array([base_uv[i], base_uv[(i + 1) % 4],
                                 roof_uv[(i + 1) % 4], roof_uv[i]], np.float32)
                if cv2.contourArea(wall) < 2.0:
                    continue
                cv2.fillPoly(img, [wall.astype(np.int32)], (124, 126, 122),
                             cv2.LINE_AA)
            cv2.fillPoly(img, [roof_uv.astype(np.int32)], (152, 154, 152),
                         cv2.LINE_AA)


def to_px(cam, pts):
    """地面坐标 -> 窗口像素。"""
    rel = np.atleast_2d(pts) - cam['origin']
    return np.stack([rel @ cam['e1'], rel @ cam['e2']], 1) * PPM


# ---------------------------------------------------------------- 地表与人

def terrain_layer(zones, origin, e1, e2, size):
    w, h = size
    rng = np.random.default_rng(SEED + 7)
    cam = {'origin': origin, 'e1': e1, 'e2': e2}

    def px_quad(q):
        return to_px(cam, q).astype(np.int32)

    # 铺底：多色调植被地面（大尺度色斑 + 细粒噪声），避免子区之间出现空洞
    coarse = gaussian_filter(rng.normal(0, 1, (h, w)), 26.0)
    coarse /= max(float(np.abs(coarse).max()), 1e-6)
    base = np.array([74, 88, 62], np.float32)[None, None, :] \
        + coarse[..., None] * np.array([18, 28, 13], np.float32)
    fine = gaussian_filter(rng.normal(0, 9, (h, w)), 2.0)
    base += fine[..., None] * np.array([0.7, 1.0, 0.6], np.float32)
    img = np.clip(base, 0, 255).astype(np.uint8)
    # 环路：依次连接各子区中心的游览道
    loop = to_px(cam, np.array([z.center for z in zones] + [zones[0].center]))
    cv2.polylines(img, [loop.astype(np.int32)], False, (150, 148, 144),
                  int(7 * PPM), cv2.LINE_AA)

    for z in zones:
        q = px_quad(z.quad)
        if q[:, 0].max() < 0 or q[:, 0].min() >= w:
            continue
        if q[:, 1].max() < 0 or q[:, 1].min() >= h:
            continue
        base, kind = TERRAIN[z.name]
        cv2.fillPoly(img, [q], base, lineType=cv2.LINE_AA)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [q], 255)
        sel = mask > 0

        if kind == 'grass':
            noise, tint = gaussian_filter(rng.normal(0, 22, (h, w)), 1.6), np.array([10, 34, 6], np.float32)
        elif kind == 'water':
            noise, tint = gaussian_filter(rng.normal(0, 9, (h, w)), 3.0), np.array([24, 6, 0], np.float32)
        elif kind == 'park':
            noise, tint = gaussian_filter(rng.normal(0, 5, (h, w)), 1.2), np.zeros(3, np.float32)
        elif kind == 'trail':
            noise, tint = gaussian_filter(rng.normal(0, 16, (h, w)), 1.8), np.array([14, 22, 10], np.float32)
        else:
            noise, tint = gaussian_filter(rng.normal(0, 7, (h, w)), 1.0), np.zeros(3, np.float32)
            grid = ((np.arange(w)[None, :] // 12 + np.arange(h)[:, None] // 12) % 2)
            img[sel] = np.clip(img[sel].astype(np.float32) + grid[sel, None] * 3,
                               0, 255).astype(np.uint8)
        px = img.astype(np.float32)
        px[sel] += noise[sel, None] + tint[None, :]
        img = np.clip(px, 0, 255).astype(np.uint8)

        if z.name in BUILDINGS:                       # 建筑屋面 + 树冠
            for _ in range(int(rng.integers(2, 5))):
                cl = z.center + z.tangent * rng.uniform(-0.25, 0.25) * z.length \
                    + z.normal * rng.uniform(-0.25, 0.25) * z.depth
                ww = rng.uniform(0.20, 0.36) * z.length
                hh = rng.uniform(0.20, 0.36) * z.depth
                bq = np.array([cl + z.tangent * ww / 2 + z.normal * hh / 2,
                               cl - z.tangent * ww / 2 + z.normal * hh / 2,
                               cl - z.tangent * ww / 2 - z.normal * hh / 2,
                               cl + z.tangent * ww / 2 - z.normal * hh / 2])
                col = int(rng.integers(120, 165))
                cv2.fillPoly(img, [px_quad(bq)], (col, col, col - 4), cv2.LINE_AA)
            for _ in range(int(rng.integers(3, 9))):
                tc = z.center + z.tangent * rng.normal(0, 0.42 * z.length) \
                    + z.normal * rng.normal(0, 0.42 * z.depth)
                p = to_px(cam, tc)[0]
                cv2.circle(img, (int(p[0]), int(p[1])), int(rng.integers(9, 20)),
                           (66, 80, 58), -1, cv2.LINE_AA)

        if kind == 'park':                            # 车位线
            for gx in np.arange(0, z.length, 8.0):
                p = z.center + z.tangent * (gx - z.length / 2)
                a1 = to_px(cam, p + z.normal * z.depth / 2)[0]
                a2 = to_px(cam, p - z.normal * z.depth / 2)[0]
                cv2.line(img, (int(a1[0]), int(a1[1])), (int(a2[0]), int(a2[1])),
                         (200, 200, 200), 1, cv2.LINE_AA)
    return img


def crowd(zone, n, w, cam, rng):
    """撒出 n 个人的真实位置，取离摄像头最近的 round(w*n) 个作为该路可见的人。

    画面中的人数因此严格等于观测算子 sum_k H[m,k]·n[k] 的取整结果。
    返回的是地面坐标，人像在屏幕空间按真实尺寸另绘。
    """
    n = int(round(float(n)))
    if n <= 0:
        return None
    gx, gy = sample_zone(zone, n, rng)
    d = np.hypot(gx - zone.center[0], gy - zone.center[1])
    k = max(0, min(n, int(round(w * n))))
    if k == 0:
        return None
    idx = np.argpartition(d, k - 1)[:k]
    return np.stack([gx[idx], gy[idx]], 1), k


def draw_people(img, xy, cols, pose):
    """把地面坐标上的人按投影画到画面上：远小近大，远的先画以形成遮挡。"""
    uv, z = pose.project(xy)
    ok = z > 1.0
    uv, z, cols = uv[ok], z[ok], cols[ok]
    jit = np.random.default_rng(SEED + 5).uniform(0.88, 1.14, len(z))
    for i in np.argsort(-z):
        hw = pose.f_px * 0.24 / z[i] * jit[i]   # 半身宽（米 × 焦距 / 深度）
        if hw < 0.55:
            continue
        cx, cy = int(uv[i, 0]), int(uv[i, 1])
        if cx < -6 or cx > IMG_W + 6 or cy < -8 or cy > IMG_H + 8:
            continue
        col = tuple(int(c) for c in cols[i])
        lin = cv2.LINE_AA if hw > 1.6 else cv2.LINE_8
        if hw > 1.1:                          # 落影，把人物压在地面上
            cv2.ellipse(img, (cx, cy + int(hw * 0.9)),
                        (max(1, int(hw * 1.3)), max(1, int(hw * 0.45))),
                        0, 0, 360, (58, 60, 56), -1, lin)
        cv2.ellipse(img, (cx, cy), (max(1, int(hw * 1.15)), max(1, int(hw * 1.85))),
                    0, 0, 360, col, -1, lin)
        if hw > 1.9:                          # 头肩：小型人像加一点暗色头影
            cv2.circle(img, (cx, cy - int(hw * 0.75)), max(1, int(hw * 0.55)),
                       (max(col[0] - 26, 0), max(col[1] - 26, 0), max(col[2] - 26, 0)),
                       -1, lin)


# ---------------------------------------------------------------- 光影与噪声

def lighting(step, day):
    hour = step * 15.0 / 60.0
    elev = np.sin((hour - 6.0) / 12.0 * np.pi)
    dl = float(np.clip(elev, 0.0, 1.0))
    b = 0.26 + 0.92 * dl ** 0.65
    warm = float(np.clip(1 - abs(hour - 12) / 6.5, 0, 1))
    tint = np.array([0.90 + 0.14 * warm, 0.95 + 0.07 * warm, 1.03 - 0.07 * warm],
                    np.float32)
    rng = np.random.default_rng(SEED + 31 * day)
    return b * rng.uniform(0.80, 1.05), tint, dl < 0.10


def post(img, bright, tint, night, rng):
    x = img.astype(np.float32) * bright * tint[None, None, :]
    if night:                                        # 夜间灯位光晕
        yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
        for _ in range(7):
            u, v = rng.uniform(0, IMG_W), rng.uniform(IMG_H * 0.45, IMG_H)
            s = rng.uniform(55, 130)
            g = np.exp(-(((xx - u) ** 2 + (yy - v) ** 2) / (2 * s * s)))
            x += (32 * g)[..., None] * np.array([1.35, 1.05, 0.75], np.float32)
    far = gaussian_filter(x, (1.4, 1.4, 0))
    ramp = np.linspace(0.34, 0.0, IMG_H)[:, None, None]
    x = x * (1 - ramp) + far * ramp                  # 远处更糊
    x += rng.normal(0, 3.4, x.shape)                 # 传感器噪声
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    vg = 1 - 0.28 * (((xx - IMG_W / 2) / (IMG_W / 2)) ** 2 +
                     ((yy - IMG_H / 2) / (IMG_H / 2)) ** 2)
    x *= np.clip(vg, 0.48, 1.0)[..., None]           # 暗角
    x = 255.0 * np.clip(x / 255.0, 0, 1) ** 1.05
    return np.clip(x, 0, 255).astype(np.uint8)


def no_signal(rng):
    f = rng.normal(28, 9, (IMG_H, IMG_W, 3))
    for _ in range(3):
        y = int(rng.uniform(0, IMG_H))
        f[y:y + int(rng.integers(2, 7))] += 26
    return np.clip(f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 主流程

def render_one(cam, zones, counts, step, day, rng):
    pose = cam['pose']
    img = cam['terrain'].astype(np.float32).copy()

    vis, cols, n_vis = [], [], 0
    for k, w in cam['targets']:
        got = crowd(zones[k], counts[k], w, cam, rng)
        if got is None:
            continue
        P, n = got
        n_vis += n
        vis.append(P)
        cols.append(CLOTHES[rng.integers(0, len(CLOTHES), n)])

    src = np.clip(img, 0, 255).astype(np.uint8)
    uv, zz = cam['pose'].project(cam['corners'])
    if (zz <= 0.5).any():
        return None, 0
    w, hgt = cam['size']                      # 单应的定义域是源图像素坐标
    src_px = np.array([[0, 0], [w, 0], [w, hgt], [0, hgt]], np.float32)
    Hm = cv2.getPerspectiveTransform(src_px, uv.astype(np.float32))
    out = cv2.warpPerspective(src, Hm, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    draw_scenery(out, cam, pose)              # 树木与建筑：提供高度线索
    if vis:                                   # 人群在屏幕空间按真实尺寸绘制
        draw_people(out, np.concatenate(vis), np.concatenate(cols), pose)
    bright, tint, night = lighting(step, day)
    return post(out, bright, tint, night, rng), n_vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', default='data/scenic_mm.pt')
    ap.add_argument('--out', default='frames')
    ap.add_argument('--probe', type=int, default=0)
    ap.add_argument('--day', type=int, default=55)
    ap.add_argument('--quality', type=int, default=88)
    a = ap.parse_args()

    d = torch.load(a.pt, map_location='cpu', weights_only=False)
    sd = d['state_dict'] if 'state_dict' in d else d
    names = list(sd['region_names'])
    areas = np.asarray(sd['areas'], float)
    H = np.asarray(sd['coverage'], float)
    n_true = np.asarray(sd['n_true'], float)
    y_vis = np.asarray(sd['y_vis'], float)
    m_vis = np.asarray(sd['m_vis'], float)
    steps = int(d.get('config', {}).get('steps_per_day', 96))

    zones = build_layout(names, areas)
    cams = [prepare(m, zones, H) for m in range(H.shape[0])]
    assert all(c is not None for c in cams), '有摄像头无法定出位姿'
    os.makedirs(a.out, exist_ok=True)

    # Set A = 测试日全天 96 步 × 12 路；Set B = 每 5 天 12:00 一帧 × 12 路
    if a.probe:
        plan = [(a.day, int(round(s))) for s in np.linspace(0, steps - 1, a.probe)]
    else:
        plan = [(a.day, s) for s in range(steps)]
        plan += [(dd, steps // 2) for dd in range(0, 60, 5) if dd != a.day]

    manifest, dev = [], []
    for i, (day, step) in enumerate(plan):
        t = day * steps + step
        counts = n_true[t]
        for m, cam in enumerate(cams):
            rng = np.random.default_rng(SEED + t * 131 + m)
            if m_vis[t, m] < 0.5:
                img, nv = no_signal(rng), 0
            else:
                img, nv = render_one(cam, zones, counts, step, day, rng)
                if img is None:
                    img, nv = no_signal(rng), 0
            fn = 'd%02d_s%02d_c%02d.jpg' % (day, step, m)
            cv2.imwrite(os.path.join(a.out, fn), img,
                        [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            clean = float(sum(H[m, k] * counts[k] for k, _ in cam['targets']))
            if m_vis[t, m] >= 0.5:
                dev.append(abs(nv - clean))
            manifest.append({'file': fn, 'day': int(day), 'step': int(step),
                             'hour': '%.2f' % (step * 15 / 60.0), 'camera': m,
                             'zone': names[cam['primary']],
                             'targets': [[names[k], round(w, 4)] for k, w in cam['targets']],
                             'valid': int(m_vis[t, m] >= 0.5),
                             'n_visible_true': int(nv),
                             'n_visible_clean': round(clean, 1),
                             'y_vis_noisy': round(float(y_vis[t, m]), 1)})
        if a.probe and i + 1 >= a.probe:
            break

    json.dump({'seed': SEED, 'image_size': [IMG_W, IMG_H], 'ppm': PPM,
               'n_frames': len(manifest),
               'visible_vs_clean_mae': float(np.mean(dev)) if dev else None,
               'frames': manifest},
              open(os.path.join(a.out, 'manifest.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('写了 %d 帧 -> %s' % (len(manifest), a.out))
    if dev:
        print('画面真值人数 vs 观测算子（未加噪）的平均绝对差 = %.3f 人' % np.mean(dev))


if __name__ == '__main__':
    main()
