# HOINet

HOINet 把景区人流风险评估拆成两个问题：潜在状态的反演，与风险概率的读出。
状态量取区域人数而非密度，使信令的空间混合观测具有可加形式；反演过程由近端
梯度法的展开迭代构造，视频、信令与锚点三条观测通道的残差反投影分别对应一条
梯度通道。

> **关于文中的章节号与公式号**：本 README 统一引用英文投稿版现稿的编号
> （§3.3 异质观测模型、§3.5 反演目标、§3.6 展开层、§4.1 数据集与数据粒度、
> §4.3 实现细节、§4.9 消融实验）。中文版另投，章节结构相近但公式编号不同，
> 请以各自稿件为准。

## 目录结构

```
src/data/       数据层：结构矩阵、观测、标签、Scenic-MM 基准
  scenario.py     Scenic-MM 基准的张量结构与读取入口（论文 §4.1）
  labels.py       式(1)(3) 密度与拥堵持续状态、阈值与风险等级
  video.py        式(9)—式(11) 带掩码的视频局部人数观测
  signaling.py    式(12)       空间混合与时间聚合的信令观测
  anchors.py      式(13)(14)   渗透率标定与锚点观测、可用性掩码
  graph.py        式(15)       功能连通图与图拉普拉斯
  dataset.py      划分训练/验证/测试并滑窗
src/model/      模型层
  hoi_net.py      HOINet 主体：展开层、三通道反投影、门控、渗透率更新
  ops.py          切比雪夫图卷积、因果时间卷积、单调读出、软阈值
  losses.py       式(28)—式(31) 训练目标
  encoder.py      视频侧的视觉编码器
  engine.py       训练循环、评估与多种子重复实验
src/evals/      评估层
  metrics.py      状态反演指标（整体/覆盖区/缺口区）与风险分级指标
  calibration.py  ECE 与 MCE 校准指标
src/config.py   配置文件到各 dataclass 配置对象的映射
configs/        实验配置
  default.yaml    论文 §4.1 与 §4.3 的规模与超参数
scripts/        命令行入口
  render_frames.py     由真值张量渲染图像通道
  train.py             训练
  evaluate.py          评估
  inference.py         推理（无标签，只产出预测）
  run_ablation.py      论文 §4.9 表 6 的消融实验
  run_robustness.py    观测缺陷实验：信令时间戳平移、渗透率扰动、锚点稀疏化。
                       论文 §4.1 只给出锚点稀疏化（100/50/25/0%）的协议，
                       另两项为仓库自带的扩展，论文未列表报告
  real_case_ucfqnrf.py 论文 §4.1 的真实图像案例：在公开点标注（UCF-QNRF 与
                       CrowdUIT，annPoints 格式相同）上实例化式(12) 与式(14)，
                       产出每张图的格点/扇区/锚点人数与 JSON
  real_case_figure.py  把上一脚本的产物拼成单幅图像案例图
  real_case_figure_stack.py  把多张单图产物纵向拼成论文图 fig:realcase
tests/          论文中可检验的不变量与结构约束
```

## 数据集

Scenic-MM 随仓库分发，位于 [`data/scenic_mm.pt`](data/scenic_mm.pt)（约 3.5 MB），
无需另行下载或生成。文件里保存基准的全部张量与结构矩阵，读取入口为
`src/data/scenario.py:load_scenario`：

```python
from src.data import load_scenario
sc = load_scenario('data/scenic_mm.pt')
```

唯一不随文件保存的是时间聚合矩阵 `time_agg`：本基准的步长与信令窗口同为
15 min，该矩阵退化为单位阵，且只取决于步长与窗口、与数据无关，故加载时由
`signaling.time_aggregation_matrix` 重建，以省去 (5760, 5760) 的存储。
论文数字中唯一无法仅凭本文件复现的是真实图像案例一节，它另外需要
UCF-QNRF 或 CrowdUIT 的原始图像与标注。

## 快速开始

```
pip install -r requirements.txt

python scripts/train.py --data data/scenic_mm.pt --epochs 100
python scripts/evaluate.py --data data/scenic_mm.pt --ckpt runs/hoinet/model.pt
```

把真值张量渲染成视频通道的图像，用于人工核对式(9)—式(11) 的观测算子与画面
是否一致，需另装可选依赖：

```
pip install -r requirements-render.txt
python scripts/render_frames.py --pt data/scenic_mm.pt --probe 4
```

默认超参数全部对应论文 §4.3：Adam，学习率 1e-4，余弦退火，批大小 8，
最多 100 轮，验证损失连续 10 轮不降即早停，5 个随机种子报告均值与标准差。

```
python scripts/train.py --data data/scenic_mm.pt --seeds 0 1 2 3 4
```

仅需验证流程是否跑通时，减少轮数并加大滑窗步长即可：

```
python scripts/train.py --data data/scenic_mm.pt --epochs 5 --stride 96
```

## 真实图像案例（论文 §4.1）

论文 §4.1 在公开数据集的点标注上实例化了一次观测模型：状态量取标注给出的
真实格点人数，观测量按式(12) 与式(14) 由该状态生成。它复现论文表中的每个数字，
需要 UCF-QNRF 或 CrowdUIT 的图像与标注（获取方式见
[`scripts/real_case_ucfqnrf.py`](scripts/real_case_ucfqnrf.py) 的文件头，仓库不
分发数据集本身。论文图 fig:realcase 的三行是一帧 UCF-QNRF 与两帧 CrowdUIT）：

```
python scripts/real_case_ucfqnrf.py --img img/img_0174c.jpg \
    --ann img/img_0174c_ann.mat --out case_a/
python scripts/real_case_figure.py --case case_a --img img_0174c --out figs/one.png
python scripts/real_case_figure_stack.py \
    --item case_a:img_0174c --item case_b:scenic_s6_153 \
    --item case_c:scenic_s7_29 --out figs/fig_realcase.png
```

`--zh` 开关出中文标注版。脚本输出的 JSON 里同时含非负最小二乘由扇区数反推状态
的结果，对应论文对该表的讨论：7 个扇区观测的总人数偏差在 3.6%—6.9% 之间，而
逐格相对误差中位数为 100%，说明观测量不足以确定状态。

## 参数说明

[`configs/default.yaml`](configs/default.yaml) 是全部参数的集中入口，取值与论文
一致；不提供配置文件时，各 dataclass 的默认值与之一致。下表按论文中的符号列出
取值与其在仓库中的位置。

### 场景规模（论文 §4.1）

| 符号 | 含义 | 取值 | 配置项 |
|------|------|------|--------|
| $N$ | 子区域数 | 18 | `scenario.n_regions` |
| $M$ | 摄像头路数 | 12 | `scenario.n_cameras` |
| — | 视频覆盖区 / 缺口区 | 12 / 6 | `src/data/video.py:gap_regions` |
| — | 摄像头有效覆盖面积占比 | 所辖子区面积的 74.2%，占景区总面积的 49.5% | `data/scenic_mm.pt` 的 `coverage` 矩阵 |
| $L$ | 信令单元（基站扇区）数 | 7 | `scenario.n_sectors` |
| $A$ | 锚点点位数 | 5（3 个出入口闸机 + 2 个人工抽样点位） | `scenario.n_anchors_gate` / `n_anchors_manual` |
| — | 锚点观测次数 | 600（60 天 × 5 个点位 × 每日 2 次） | `src/data/anchors.py` |
| — | 锚点占区域-时刻比 | 0.58% | 同上 |
| — | 设备渗透率 | 0.36 – 0.52，均值 0.45 | `src/data/anchors.py:estimate_penetration` |
| $E$ | 外部变量维度 | 8 | `scenario.n_ext` |
| $K$ | 风险等级数 | 4（低/中/高/极高） | `scenario.n_classes` |
| — | 类别占比 | 实测 44.0% / 30.9% / 17.9% / 7.2% | 由 `data/scenic_mm.pt` 的真值等级 `risk` 统计得出 |
| — | 区域-时刻总数 | 5760 × 18 = 103 680 | `scenario.days` × `scenario.steps_per_day` × `scenario.n_regions` |
| — | 连续观测天数 | 60 | `scenario.days` |
| — | 视频采样间隔 | 60 s | `src/data/video.py` |
| — | 信令聚合窗口 | 15 min | `model.window_minutes` |
| $T$ | 序列长度 | 96（15 min 步长，对应 24 h） | `model.seq_len` |

这些取值记录在 `data/scenic_mm.pt` 的 `config` 段，由
[`src/data/scenario.py`](src/data/scenario.py) 读成 `ScenarioConfig`。

> 类别占比一栏由数据集里的真值等级直接统计得出，不再涉及任何目标比例参数。

### 模型结构（论文 §3.6 与 §4.3）

| 符号 | 含义 | 取值 | 位置 |
|------|------|------|------|
| — | 视觉编码器卷积核 | 3×3 / 5×5 / 7×7 | `VisualConfig.branch_kernels` |
| — | 视觉编码器通道数 | 64 / 128 / 256 | `VisualConfig.branch_channels` |
| — | 三分支输出空间尺寸 | 输入的 1/4、1/8、1/16 | `src/model/encoder.py` |
| $d_v$ | 融合后视觉特征维度 | 256 | `VisualConfig.d_visual` |
| $N_s$ | 每个基本步内抽取的视频子步数 | 30 | `src/data/video.py` |
| $d$ | 反演层嵌入维度 | 256 | `model.d_embed` |
| $K$ | 切比雪夫阶数 | 3 | `model.cheb_k` |
| $K_1, K_2$ | 时间卷积核大小 | 3, 5 | `model.tcn_kernels` |
| $S$ | 展开阶段数 | 4 | `model.n_stages` |
| — | 空间传播与时间补全算子 | 参数在 $S$ 个阶段间共享 | `src/model/ops.py` |
| $d_\pi$ | 渗透率低秩维度 | 8 | `model.pi_rank` |
| $d_e$ | 环境特征维度 | 8 | `HOINetConfig.n_ext` |
| — | 风险读出头 | 线性-softmax，无隐藏层（式(6)） | `src/model/ops.py:MonotoneReadout` |

视觉分支的参数集中在 `src/model/encoder.py:VisualConfig`，不在 `configs/default.yaml`
的 `model` 段内，修改需直接改该 dataclass。

### 训练设置（论文 §4.3）

| 含义 | 取值 | 配置项 |
|------|------|--------|
| 优化器 | Adam | `src/model/engine.py` |
| 初始学习率 | $1 \times 10^{-4}$ | `train.lr` |
| 学习率调度 | 余弦退火 | `train.eta_min_ratio` |
| 批大小 | 8 | `train.batch_size` |
| 训练轮数 | 100 | `train.max_epochs` |
| 早停耐心值 | 10 | `train.patience` |
| 随机种子 | 5 个，报告均值 ± 标准差 | `scripts/train.py --seeds` |

论文报告的运行环境为 NVIDIA RTX 4090（24 GB）、Python 3.9 与 PyTorch 2.1.0。
本仓库只依赖 `torch` 与 `PyYAML`，全部脚本在 CPU 上即可运行。

### 损失权重

| 论文符号 | 含义 | 取值 | 配置项 |
|----------|------|------|--------|
| $\lambda_r$ | 式(28) 风险分类损失权重 | 0.1 | `loss.lambda_risk` |
| $\lambda_o$ | 式(28) 观测一致性权重 | 0.05 | `loss.lambda_obs` |
| $\lambda_{\mathrm{con}}$ | 式(28) 跨模态一致性权重 | $1 \times 10^{-4}$ | `loss.lambda_cons` |
| $\beta$ | 式(29) 密度项系数 | 1.0 | `loss.beta` |
| $\lambda_{\mathrm{rel}}$ | 式(29) 相对误差项权重 | 0.1 | `loss.lambda_rel` |
| $\varepsilon_0$ | 式(29) 相对误差分母的稳定项 | $10^{-3}$ | `loss.eps` |
| $\lambda_{\mathrm{ord}}$ | 式(30) 序数损失权重 | 0.1 | `loss.lambda_ord` |
| $\varepsilon$ | 式(24) 预条件阻尼系数（论文中记作 $\varepsilon$，与上一行的 $\varepsilon_0$ 是两个不同的量） | $10^{-6}$ | `model.eps_curv` |

## 消融与鲁棒性实验

```
python scripts/run_ablation.py --data data/scenic_mm.pt \
    --seeds 0 1 2 3 4 --internal --out runs/ablation
python scripts/run_robustness.py --data data/scenic_mm.pt \
    --seeds 0 1 2 3 4 --out runs/robustness
```

消融脚本按论文消融表的勾选列逐行设置 `use_*` 开关，报告准确率与宏平均 F1；
鲁棒性脚本覆盖信令时间戳平移、渗透率扰动与锚点稀疏化三类缺陷，
三者都只在测试区间注入缺陷，训练与验证区间保持干净。其中锚点稀疏化的四档
（100% / 50% / 25% / 0%）与论文 §4.1 所述协议一致，另两类缺陷的数值论文未报告。

## 测试

```
python tests/test_invariants.py     # 也可用 pytest tests/ 收集
```

## 式(21) 的预条件

论文式(21) 给出的是数据项梯度，其量级由观测精度 `Omega^2` 决定，比待修正
的状态量小四个到十个数量级；式(25) 只有一个可学习步长供三条通道共用，无法
同时补偿三个不同的量级，故预条件是把式(21) 与式(25) 接通的前提。

实现取带阻尼的完整 Hessian 求解 `(Hessian + lambda I)^{-1} g`，即式(23) 与
式(24)，其依据是式(21) 的梯度恒正交于观测算子的零空间，因而求解结果在可
辨识子空间内是精确的牛顿步、在零空间内为零。`HOINetConfig.precond` 提供
三种方式：

```
precond: full    # 默认，完整求解
precond: diag    # 对角（Jacobi）近似，作对照
precond: none    # 不做预条件，即式(21) 原样
```

对角近似在稠密观测算子下偏差较大——信令通道的 `B` 为 (7, 18) 且列和为 1，
对角近似会把该通道放大约 10 倍，完整求解则落在参照量的 0.999 倍。

## 引用

若本仓库对您的工作有帮助，请引用论文原文。
