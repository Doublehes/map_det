# model_slim2 工作总结

> **最新进展见 §12（2026-10-16）。** §8 / §10.9 / §11.6 的遗留项已按 §12 标注状态。

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

1. **长度自适应（该长则长、该短则短）** —— **部分推进（见 §12）**
   - 现状：`span`/`arc` 是"全局统一"先验（所有线越长越好），但正确长度应由区域结构决定
   - 已确认场景：用户要的"应该短"主要指**被非可通行区截断（尽头）**的情形
   - **已做**：关键点覆盖率损失（§12.3）提供了一种"覆盖"信号；**方案 C 骨架覆盖仍未实现**。
   - **已定未实施**：§12.5「车后起点锚定」，让线从车后沿车道贯穿到车前 = 越长越好。
2. **朝向**：当前不做矫正（2D 原生对齐）；若上游分割图朝向变化需再议 —— **未变化**
3. **部署**：推理输入若为上游模型的**预测分割**（含噪声），需增加输入噪声增广提升鲁棒性 —— **未做**
4. **权重调参**：`repel_margin`、各项权重需结合训练效果微调 —— **未变化**；§12 新增 `keypoint_coverage_weight`、（拟加）`start_anchor_weight` 亦待调。

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
1. **尚未成"长条"**：`span` 恰好停在阈值 0.4 ⇒ 当前是"刚好达标"的短直段，而非贯穿区域的长线。想更长可调 `min_span`/`span_weight`，或引入**覆盖项**。 —— **仍成立**；且 §12.5 定位到新现象"关键点只在前半区 ⇒ 线挤在前半区、后排空"。
2. **长度自适应/输入相关**未根治：输出已随输入变化，但仍以自监督全局先验为主；**方案 C（骨架/软带覆盖 + Chamfer）** 尚未实现。 —— **仍成立**（已新增关键点覆盖损失，方案 C 未做，见 §12.3 / §12.5）。
3. `summary.md` §5 表格为历史记录，最新以本节为准。

---

## 11. 迭代总结（2026-10-15）

### 11.1 可通行区距离场重构（独立调试脚本）
- **旧场问题**：`clip(dist/max_dist_px)` 宽路中心一片=1（平台）、窄路中心<1（宽路偏置）。
- **新场（方案A，窗口归一化）**：开运算(去 < `min_road_width` 的窄结构) → `dist=距离变换`
  → `local_max=灰度膨胀(dist, ridge_window)` → `field=clip(dist/local_max,0,1)`。
  性质：中心恒=1、宽路衰减慢/窄路快、无平台；非可通行=0。
- **独立脚本 `model_slim2/vis_dist_field.py`**（不接模型）：面板 `seg | old dist | window field | window+removed`
  + 扫描行剖面图；交互 `↑/↓(w/s)` 移动扫描行、`←/A →/D/空格` 切帧、`ESC/Q` 退出。
- 试过“最大内切圆(disk)”方案（解决宽窄交界污染）→ **效果不明显，仅删除磁盘法**，保留窗口法。

### 11.2 骨架提取（scikit-image）
- 安装：`pip install --no-deps scikit-image tifffile lazy_loader`；**核验 torch/numpy/scipy 未变**。
- `build_skeleton` 流水线：开运算(`skel_min_width`) → 高斯平滑(`smooth`) → **只填小洞(`fill_max_area`)**
  → 最大连通域 → `skeletonize` → 剪毛刺(`prune_len`) → 最大骨架连通分量。
- 第 5 面板叠加：**绿=完整骨架 / 红=最长路径**；CLI `--skel-min-width/--smooth/--prune/--fill-max-area`
  （与距离场 `--min-width` **解耦**）。
- **修复“骨架零散”**：开阈值太小 → 交叉点多（如 tok0 47→1，提阈值+剪枝+平滑+填洞）。
- **修复“骨架穿类别 0”**：根因是**闭运算**（把 ≤核宽的非可通行缝隙糊上，连通两块可动区）→ **移除闭运算**；
  残留（`fill_holes` 把被可动区包围的类别0口袋也填掉）→ 改为**只填面积 ≤ `fill_max_area`(8) 的小洞**。

### 11.3 距离场移植进训练
- `dataset.py` 新增 `mask_to_dist_field_norm(...)`（**独立实现**，与 `vis_dist_field.build_field` 逐位一致，
  实测最大差=0）；**保留**原 `mask_to_dist_field`（`dist_field_mode='old'` 仍可用）。
- `config.data` 新增：`dist_field_mode='new'`（默认）、`dist_min_road_width=5.0`、`dist_ridge_window=30.0`。

### 11.4 初始化方向改为“自下而上”
- `head.init_weights`：`v = torch.linspace(0.9, 0.1, P)`（首点在下=车后，末点在上=车前）；
  验证首点 `(0.5,0.9)`、末点 `(0.5,0.1)`。

### 11.5 多线实验：低端聚合 + 顶端发散（exp3, N=3）
- 目标：N=3 线（直行/左转/右转）；**底端共享车后起点、顶端发散表示不同岔路**。
- 新增损失：`start_gather`（各线起点 `p0` 两两距离，趋 0=聚合）、
  `diverge`（各线终点 `p_last` 相关排斥=发散）。
- 接线：`head.loss/loss_breakdown`；`infer` per-pair 热力图 `repel | spread | start_gather | diverge`。
- 配置：`default` 两项权重=0（`diverge_margin=0.2`）；`exp3` 设
  `num_queries=3, start_gather_weight=1.0, diverge_weight=1.0, diverge_margin=0.2, spread_weight=0`。
- 初始态（3 query 完全相同）：`start_gather=0`、`diverge=0.2`。
- 短训（1440 步，无 NaN）：**底端 `p0` 均距≈0.003**（聚合）、**顶端 `p_last` 均距≈0.158**（发散）。
- **重设计 `diverge`**：由“终点↔终点距离”改为 **“终点 → 其它线全部点的最小距离”**
  （点-点 min，免去点到线段；16 点密集，误差≈半段长）。单元验证：终点压在别线上→被罚、远离→0、对角=0。
- **未达标**：重设计后“终点到别线的最小距离（最差对）”仅 **0.022**（期望 ≥~0.1）→ 分散**缓解未根除**。

### 11.6 遗留 / 后续（状态更新见 §12）
1. **顶端分散不够**：可选加大 `diverge_weight`/`diverge_margin`、延长训练，或改为**顶端 k 点**排斥，
   或回到"点到线段"。 —— **未解决**（今天未处理；§12.4 的 per-query `u` 铺开可作弱先验）。
2. **单路图**：多余线会被迫摊开（与 `interior/cost` 冲突）→ 需**有效性 C1（最近线覆盖）/骨架分支 C4/方向约束**。 —— **部分解决**：已加**关键点覆盖**损失（§12.3，一种覆盖机制）。
3. **方向约束**未加（靠初始化先验；`p_last` 视为顶端，翻转风险仍在）。 —— **仍成立**（§12.4 per-query `u` 初始化 + §12.5 起点锚定，属弱方向先验）。
4. **骨架"穿类别 0"**：`vis_dist_field.py` 全库仍 23/644 有 ≤3px 越界（源于平滑+小开阈值）；
   是否加严格兜底 `skel &= 可动区` 未定；**该脚本未接入训练**。 —— **仍成立**（今天未动骨架）。
5. `interior_weight` 是否需因**新距离场分布变化**而重调，未定。 —— **更需重调**：距离场已改为"只保留最大连通域"（§12.2），分布变了。

---

## 12. 迭代总结（2026-10-16）

### 12.1 关键点提取改用窗口场（`vis_dist_field`）
- `extract_keypoints_lm` 第 1 参数由 `fg_open` 改为**已算好的归一化窗口场 `win_f`**（不再内部重算
  `distanceTransform`）；前景判定 `field>0`。
- 参数语义调整：`KP_EPS 0.5(Xpx) → 0.02(字段单位)`；`KP_DIST_FLOOR 2.0px → KP_FIELD_FLOOR 0.5`；
  关键点圆半径固定 3；CLI `--kp-dist-floor → --kp-field-floor`。
- 验证：前 3 张图关键点落在路中心脊线（u≈0.52）、`f∈[0,1]`、按 u 排序、仅 `v≤0.7`。

### 12.2 距离场只保留最大可通行连通域
- `vis_dist_field.build_field`：**开运算之后**加"只保留最大连通域（8 邻接）"，距离变换只在主区域；
  返回值改为 `(field, fg_main, removed, removed_cc)`；`window+removed` 面板 **红=开运算移除 /
  洋红=非最大连通域移除**；CLI 加 `--no-keep-main`、`--cc-connectivity{4,8}`；`profile_row` 改用主区域。
- **同步到训练**：`dataset.mask_to_dist_field_norm` 新增独立 `_largest_cc`，开运算后固定
  `keep_main(8 连通)`；**不加任何配置项**（签名/调用不变）。
- 验证：`dataset` 场与 `vis_dist_field.build_field(keep_main=True, conn=8)` **逐位一致（max diff=0）**。

### 12.3 关键点覆盖率损失接入训练
- `dataset`：`__getitem__` 复用已算好的窗口场提取关键点（独立实现 `extract_keypoints_lm`），
  输出定长 `keypoints(K,2)+keypoints_valid(K,)`（K=`kp_max`，补 0）；`collate_fn` 堆叠。
- `losses`：`keypoint_dist_per_kp`（关键点→**全部预测点**最近点距，纯点到点）、
  `keypoint_coverage_per_kp = relu(d−thr)`、`keypoint_coverage_loss`（有效关键点均值）、
  `keypoint_coverage_rate`（覆盖率指标）。
- `head`：`keypoint_coverage_weight/thr`，`loss()` 附 `keypoint_coverage`（键恒在）；
  `loss_breakdown` 可选附 per-kp。`model.compute_loss` 透传 `keypoints/keypoints_valid`。
- `config`：`data` 加 `kp_*`（`kp_dmin` 实为 0.05）；`map_det_head` 加
  `keypoint_coverage_weight=0 / thr=0.05`；新增 **exp4 / exp5 / exp6**（N=3，覆盖率权重 2/5/5）。
- `train.run_val`：加 `val_kp_coverage`。
- `infer`：加**第 5 栏「keypoints」面板**（压暗分割 + 预测线/点 + `v_front` 分界线 + 关键点黄圈+序号），
  header 显示 `kp=… cov=…`。
- 验证：exp4 smoke 前向/反向正常、`loss_dict` 含 `keypoint_coverage`、无 NaN；`infer --save` 5 栏面板正常。

### 12.4 head 初始化为 per-query 参考线
- 现象：`u=torch.full((P,),0.5)` + `reference_points_embed.weight=0`（共享 bias）⇒
  **N 条 query 的参考线完全相同**。
- 改法（`Slim2Head.init_weights`，不动基类）：`u=linspace(0.2,0.8,N)`（N=1→0.5）、`v=linspace(0.9,0.1,P)` 共享；
  用 **pinv 精确拟合** `W=(pinv(query_embedding.weight)@L).T`、bias=0，使 `ref_points=sigmoid(Q@Wᵀ)` 恰好=目标线；
  保留 `query_embedding` 随机初始化；`assert N<=D`；`reg_branches` 保持中心先验 `u=0.5`。
- 验证：N=3 → `u=[0.2,0.5,0.8]`（`|du|≈3e-8`）、跨 query `std=0.3`；N=1 与现状一致。

### 12.5 遗留：线偏前（方案已定，未实施）
- **现象**：关键点仅车前（`v≤0.7`）→ 覆盖率**单向**（关键点→点）把线吸在车前；`span/arc` 只是**下界**、
  无"变长"力 ⇒ 线挤在前半区、后排空（"起点都在前半区"）。
- **实测**（8 图）：可通行区 `v` 延伸到 0.72~1.00，但 `v>0.7` 的关键点被 `v_front` 丢弃。
- **已定方案**：**保留车前关键点语义**，**新增「车后起点锚定」损失**——锚点=**可通行区后端脊线点**
  （数据驱动，`field≥0.9` 中 `v` 最大者）；`relu(‖p0−anchor‖−margin)`；与 `start_gather` 叠加
  ⇒ 线从车后沿车道贯穿到车前（越长越好）。**未实施**。
- 其它候选（未采纳）：关键点扩到全域 / 长度奖励 / 骨架覆盖（方案 C）。

