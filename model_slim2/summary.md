# model_slim2 工作总结

## 1. 任务目标

**在 2D BEV 分割图上做线检测**，输出矢量化的道路中心线 / 指引线（可引导车辆行驶）。

- **输入**：一张 BEV 语义分割图（`data_seg/rt_bev_semantic/*.png`）
- **输出**：若干条矢量线，**每条线 = 16 个 `(u, v)` 归一化坐标点**（图像坐标系，u→宽、v→高）
- **监督方式**：**自监督**——没有任何人工标注的线 GT，模型先预测线，再把这些线采样到"代价图 / 距离场"上计算损失

这是一个纯 2D 任务，不涉及 3D / BEV 米制坐标，因此 config 中**没有** `pc_range` / `roi_size` / `canvas_size`，也**不做**朝向矫正（输入与输出共享同一图像坐标系）。

---

## 2. 目录结构

```
model_slim2/
├── config.py     # 配置 (AttrDict)
├── dataset.py    # 分割图 → one-hot 输入 + 同源代价图/距离场
├── model.py      # ResNet18(5入通道) + FPN + Slim2Head
├── head.py       # Slim2Head(MapTRHead 子类), 重写 loss
├── losses.py     # 自监督损失 (未聚合 + 聚合 + 占比)
├── train.py      # 训练循环
├── infer.py      # 推理可视化 + loss 明细
└── __init__.py
```

---

## 3. 数据与输入

- **数据源**：`data_seg/rt_bev_semantic`（644 张 PNG，256×160，标签值 `{0,1,2,3,255}`）
- **输入表示**：**one-hot 5 通道**（4 类 + 1 个 ignore 通道），不 resize、不矫正朝向
- **同源监督信号**（从同一张标签图派生，保证与输入对齐）：
  - **代价图 cost**：类别查表 `{0:1.0, 1:0.5, 2:0.05, 3:1.0}`，ignore=1.0，高斯平滑（sigma=3px）软化边界
  - **距离场 dist_field**：可通行区（`drivable_classes`）到边界的距离变换，归一化到 [0,1]
- **几何增强**：flip / 小角度旋转 / sin 弯曲，**只作用于标签图**，再派生所有信号（自监督下增强"免费"）
- **切分**：固定种子 9:1 切 train/val

---

## 4. 模型架构

```
分割图 (5ch) → ResNet18(conv1改5入通道) → FPN(P2) → Slim2Head → (B, N, 16, 2)
```

- **Backbone**：ResNet18，`conv1` 输入通道 3→5（原权重取均值复制）
- **Neck**：FPN，取 P2（stride4），out_channels=128
- **Head**：`Slim2Head` **继承 `MapTRHead`**，复用其 query / reference points / transformer decoder / cls+reg 分支，**只重写 `loss`**
  - `MapTRHead` 输出 `reg_preds (B, N, 16, 2)`，经 `sigmoid` 得到 [0,1] 归一化 `(u,v)`，正是所需
  - 自监督下 `cls_scores` 不参与损失（推理时用代价过滤），结构保留

---

## 5. 自监督损失设计（历史设计，最新见 §10）

> **注**：本节为早期 8 项设计的记录；当前实际损失集与最新迭代见 **§10 今日迭代总结（2026-10-14）**。

**核心思路**：无线 GT，用"代价图 + 距离场"把预测线放到可通行区域中心。
**前 2 项是"正确信号"，后 6 项是"防退化的正则"**（单用前 2 项会塌缩到平凡解）。
所有项均为**非负惩罚（≥0，越大越差）**。

| 损失 | 形式 | 作用 | 防的退化 |
|---|---|---|---|
| `cost` | `mean(cost@点)` | 点落低代价（可通行）区 | 线进障碍区 |
| `interior` | `mean(1 - dist@点)` | 线趋向区域几何中心（≈骨架） | 线贴可通行区边缘 |
| `smooth` | 二阶差分模长均值 | 线平滑 | 锯齿、乱序 |
| `repel` | 点到另一线折线距离 < margin 的 hinge | 线间拉开 | N 条线塌缩成一条 |
| `arc` | 弧长 Σ‖Δp‖ < min_arc 的 hinge | 线够长（真实长度） | 线缩成一个点 |
| `spread` | query 输出余弦相似度 > thr 的惩罚 | 形状多样 | 多线输出雷同 |

**关键点**：
- `cost` / `interior` 用 `grid_sample` 在归一化坐标上采样地图，对预测坐标**可微**
- 距离场（`dist_field`）的脊线 ≈ 骨架，`interior` 用软吸引子把线拉向中心，无需硬骨架化

---

## 6. 关键迭代与决策

### 6.1 `repel` 升级：点到折线 + 稠密聚合
- **问题**：原"点到离散点 + min 聚合"在**一长一短重合**时，长线采样稀疏 → 点对距离偏大 → 漏检；且 min 只让最近一对点产生梯度（稀疏）
- **改法**：`point_to_polyline_dist`（点到**线段**距离，与采样密度无关）+ 对所有越界点对**取均值**（稠密梯度）
- **验证**：共线重合 0.0275→0.0495、短压长曲线 0.0447、不重合仍为 0

### 6.2 非负改写 + 贡献占比日志
- **问题**：`interior` 曾是"负损失"（奖励取负），与正的正则项**符号相反、量纲不同**，难以比较权重
- **改法**：
  - 非负改写（梯度**完全等价**）：`interior: -mean(dist) → mean(1-dist)`
  - 新增 `loss_shares(loss_dict)`：各项占总损失的百分比，直接看"谁在主导"
- **验证**：8 项全 ≥0、占比之和=1、梯度逐位等价

### 6.3 loss 明细接口 + 可视化
- `head.loss_breakdown()` 返回**未聚合**的逐项：
  - per-query `(B, N)`：`cost` / `interior` / `smooth` / `spacing` / `arc` / `span`
  - per-pair `(B, N, N)`：`repel` / `spread`
- `infer.py`：
  - 控制台打印 per-query 表格（每条线的各损失）
  - per-pair 热力图（`repel`/`spread` 的 N×N），交互模式开**第二个窗口**，`--save` 模式存 `{token}_loss.png`

---

## 7. 验证结果

- **数值**：各聚合函数与改写前逐位等价；梯度等价
- **非负/占比**：各项 ≥0，share 之和 = 1
- **训练 smoke**：各损失稳定下降、占比日志正常
- **可视化**：四联面板（分割上色 | 代价图 JET | 距离场 VIRIDIS | 预测线叠加）正常

---

## 7.1 验证指标说明（`run_val`，自监督无 GT）

`train.py` 的 `run_val` 汇总所有预测线 `(B, N, P, 2)` 的归一化 `(u,v)` 坐标，
分两类指标（详见函数 docstring）：

**一、正确性指标（依赖监督信号）**

| 指标 | 定义 | 方向 |
|---|---|---|
| `val_cost` | 预测点采样代价图的 Lp 均值 | 越低越好（点落在可通行区） |
| `val_interior` | 预测点采样距离场的均值（距离场=到边界距离，脊线处最大） | 越高越好（越靠中心） |

**二、几何诊断指标（只看形状，无线 GT）**
记一条线的 P 个点为 `p0..p15`，相邻段长 `s_i = ‖pᵢ₊₁ − pᵢ‖`：

| 指标 | 定义 | 含义 / 理想值 |
|---|---|---|
| `val_arc` | `mean(Σ s_i)` | 真实弧长（折线总路径长度） |
| `val_span` | `mean(‖p15 − p0‖)` | 净跨（首尾端点直线距离），应与 arc 相近 |
| `val_arc_span` | `mean(arc / span)` | 缠绕度：1=笔直，~1~2=圆滑弯道，十几=打结/绕圈/塌成小团 |
| `val_turn` | `mean(mean(1 − cosθ))` | 平均转角（θ=相邻段夹角）：0=全程直，2=每步掉头，越小越顺 |
| `val_spacing_cv` | `mean(std(s)/mean(s))` | 等距度（段长变异系数，尺度无关）：0=完全等距 |

> 例：直线等距 → `arc/span≈1, turn≈0, spacing_cv≈0`；闭环/小团 → `arc/span` 十几；
> 圆弧 → `arc/span` 略大于 1（半圆≈1.57）。
> `arc/span/turn/spacing_cv` 均在归一化坐标上计算，用于训练期快速判断线是在
> “敞开、顺畅、等距”，还是打结/塌缩/输入无关（跨样本方差为 0）。


---

## 8. 待办 / 未解决问题

1. **长度自适应（该长则长、该短则短）** —— **已讨论、暂缓**
   - 现状：`span`/`arc` 是"全局统一"先验（所有线越长越好），但正确长度应由区域结构决定
   - 已确认场景：用户要的"应该短"主要指**被非可通行区截断（尽头）**的情形
   - 选定方案：**方案 C 骨架覆盖监督**（离线对可通行区骨架化 → 用 Chamfer/双向覆盖损失让线贴合骨架，长度由骨架分支天然决定）——**尚未实现**
2. **朝向**：当前不做矫正（2D 原生对齐）；若上游分割图朝向变化需再议
3. **部署**：推理输入若为上游模型的**预测分割**（含噪声），需增加输入噪声增广提升鲁棒性（目前未做）
4. **权重调参**：`repel_margin`、各项权重需结合训练效果微调

---

## 9. 使用方法

```bash
# 训练 (config 内可调超参)
conda run -n workspace python model_slim2/train.py

# 推理可视化: 四联面板 + 第二窗口显示 per-pair 热力图 + 控制台打印 loss/占比
conda run -n workspace python model_slim2/infer.py

# 批量导出 (每样本存 {token}.png 与 {token}_loss.png)
conda run -n workspace python model_slim2/infer.py --save ./vis_slim2 --start 0
```

**交互按键**：`A/←` 上一张，`D/→/Space` 下一张，`ESC/Q` 退出

**checkpoint**：`{work_dir}/latest.pth`（默认 `./work_dirs/slim2`）

**关键配置**（`config.py`）：
- `data.cost_values`：类别→代价映射
- `data.drivable_classes`：可通行类（决定距离场）
- `map_det_head.num_queries / num_points`：线数与每条线点数
- `map_det_head.*_weight`：各项损失权重（用 `infer` 的占比日志辅助平衡）

---

## 10. 今日迭代总结（2026-10-14）

### 10.1 问题诊断（模型输出"绕成一团"）
- **现象**：训练后输出线**绕成一团**；`smooth` 看似在下降却拦不住打结。
- **实测证据**：加载 checkpoint，对 15 张验证图统计——跨样本首点 `std(u,v)≈[0.007,0.003]`、`arc≈0.404`、`span≈0.045` ⇒ **所有输入输出同一个固定小团**（输入无关的**模式坍塌**）。
- **根因**：
  1. `length` 用**总弧长**（原地振荡即可凑够，可被伪造）；
  2. `smooth` 尺度混乱（`×6.4/×10`）+ 步长项"偏离自身均值"，对**均匀小团免费**；
  3. `interior` 是**点吸引子**，且没有任何"覆盖/输入相关"约束；`reference points` 与输入无关。

### 10.2 损失重构（方案①：真实弧长 + 曲率 + 等距 + Lp）
- **有效项（6）**：`cost / interior / smooth / spacing / arc / span`；`repel/spread` 关闭（单线为 no-op）。
- `smooth`：**转向** `Lp(1−cosθ)` + **最小步长** `Lp(relu(min_step_px−‖Δp‖))`，用各向同性像素坐标 `[w,h]`（避免宽高比扭曲角度）。
- `spacing`（新）：段长相对偏差 `Lp(|s−mean(s)|/mean(s))`，**尺度无关**，保证等距。
- `arc`：弧长下界 `relu(min_arc − Σ‖Δp‖)`（**真实长度**）。
- `span`（新）：端点距下界 `relu(min_span − ‖p_last − p_first‖)`，**逼线张开**、防紧凑团/环（弧长可伪造、端点距不能）。
- **线内逐点/逐段聚合统一用 Lp 范数（`lp_p=4`）**：放大"最差点/段"，避免均值把难点平均掉（仅线内维度；batch/query 仍用均值）。
- **术语统一**：损失侧 `length` → `arc`（配置键 `arc_weight` / `min_arc`，与指标 `val_arc` 一致）。

### 10.3 关键 Bug 修复
- **NaN（训练中期突变）**：`lp_reduce(x) = (mean(xᵖ))^(1/p)` 在 **x≡0**（线变直/段长变均匀）时外层 `^(1/p)` 梯度无穷 → 权重被污染成全 NaN。**修复**：根号内加 `eps`。
- `smooth`/`spacing` 改用**平滑段长** `sqrt(‖v‖²+ε²)`，`spacing` 分母 `detach` → 近 0 段梯度有界。
- 验证：修复后连续 3960 步无 NaN（原崩溃点 ≈1829）。

### 10.4 删除
- 删除 `grow`、`order` 两项损失，及关联的 `ARC_MAX`、配置键、文档（`head/loss/infer/summary` 全部清理，`grep` 无残留）。

### 10.5 数据切分修复
- `val_ratio` 由 `0.9` 改为 `0.1`：**train 580 / val 64**（此前误配成 train 仅 65 / val 579）。

### 10.6 初始值敏感（偶发塌进局部极小）修复
- **根因**：`reference_points_embed` 与 `reg_branch` 末层均为 `xavier` ⇒ 输出≈0 ⇒ `sigmoid≈0.5` ⇒ **16 个点初始叠在图像中心的团**；能否"炸开"取决于随机轨迹 ⇒ 同配置 3 次可能坏 1 次。
- **修复**：新增 `Slim2Head.init_weights()`——在基类 xavier 之后，把上述两层**权重置 0、bias 设为先验竖直线 `u=0.5, v=0.1→0.9` 的反 sigmoid**，使初始输出即为一条竖向直线，起步就满足 `arc/span/spacing/smooth`，绕开塌缩盆地。（只改 `model_slim2/head.py`，不动基类）
- **验证**：3 次不同随机初值短训 **3/3 收敛**——`arc/span≈1.01`、`span≈0.40`、`turn≤0.015`、无 NaN；且 `std(u)≈0.12~0.15` 表明输出**随输入变化**（不再是固定线）。
- **效果（实测反馈）**：修复后**训练效率与训练效果均显著提升**——收敛更快、输出更稳定合理。
  原因：起点即落在合理区域（初始就满足 `arc/span/spacing/smooth`），省去挣脱"中心团"
  点吸引子的大量步数；且注意力采样点初始沿直线铺开，decoder 学习更快；同时绕开塌缩盆地，
  run 间一致性提高。

### 10.7 验证指标（见 §7.1）
- `run_val` 新增几何诊断：`val_arc / val_span / val_arc_span / val_turn / val_spacing_cv`。

### 10.8 当前有效配置（`default`）
```
num_queries=1, num_points=16
cost_weight=0.0, interior_weight=1.0, smooth_weight=2.0,
spacing_weight=1.0, arc_weight=1.0(min_arc=0.4), span_weight=1.0(min_span=0.4)
smooth_min_step_px=2.0, lp_p=4.0
repel_weight=0.0, spread_weight=0.0
```

### 10.9 遗留 / 后续
1. **尚未成"长条"**：`span` 恰好停在阈值 0.4 ⇒ 当前是"刚好达标"的短直段，而非贯穿区域的长线。想更长可调 `min_span`/`span_weight`，或引入**覆盖项**。
2. **长度自适应/输入相关**未根治：输出已随输入变化，但仍以自监督全局先验为主；**方案 C（骨架/软带覆盖 + Chamfer）** 尚未实现。
3. `summary.md` §5 表格为历史记录，最新以本节为准。

