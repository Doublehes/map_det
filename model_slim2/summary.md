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

## 5. 自监督损失设计（8 项）

**核心思路**：无线 GT，用"代价图 + 距离场"把预测线放到可通行区域中心。
**前 2 项是"正确信号"，后 6 项是"防退化的正则"**（单用前 2 项会塌缩到平凡解）。
所有项均为**非负惩罚（≥0，越大越差）**。

| 损失 | 形式 | 作用 | 防的退化 |
|---|---|---|---|
| `cost` | `mean(cost@点)` | 点落低代价（可通行）区 | 线进障碍区 |
| `interior` | `mean(1 - dist@点)` | 线趋向区域几何中心（≈骨架） | 线贴可通行区边缘 |
| `smooth` | 二阶差分模长均值 | 线平滑 | 锯齿、乱序 |
| `repel` | 点到另一线折线距离 < margin 的 hinge | 线间拉开 | N 条线塌缩成一条 |
| `length` | 端点距 < min_length 的 hinge | 线够长 | 线缩成一个点 |
| `spread` | query 输出余弦相似度 > thr 的惩罚 | 形状多样 | 多线输出雷同 |
| `order` | 相邻点 u 增量为负的 hinge（弱） | u 单调 | 自交叉、回折 |
| `grow` | `mean(1 - 加权端点距/SPAN_MAX)` | 促线横跨走廊 | 线过短 |

**关键点**：
- `cost` / `interior` 用 `grid_sample` 在归一化坐标上采样地图，对预测坐标**可微**
- 距离场（`dist_field`）的脊线 ≈ 骨架，`interior` 用软吸引子把线拉向中心，无需硬骨架化

---

## 6. 关键迭代与决策

### 6.1 `repel` 升级：点到折线 + 稠密聚合
- **问题**：原"点到离散点 + min 聚合"在**一长一短重合**时，长线采样稀疏 → 点对距离偏大 → 漏检；且 min 只让最近一对点产生梯度（稀疏）
- **改法**：`point_to_polyline_dist`（点到**线段**距离，与采样密度无关）+ 对所有越界点对**取均值**（稠密梯度）
- **验证**：共线重合 0.0275→0.0495、短压长曲线 0.0447、不重合仍为 0

### 6.2 `grow` 增长激励（一波三折）
- **动机**：`length` 是单边 hinge，线长超过 min_length 后**无增长激励**，长度卡在阈值附近
- **第一版**：按需求用"**求和弧长** × 可通行权重" → 实测**严重退化**：线在走廊里**锯齿状来回摆动**刷弧长（弧长 3.31，直线最多 ~1.4）
- **第二版**：改用"**端点距（span）** × 可通行权重" → 直线最大化、摆动反而减小，无钻空子空间（弧长回到 ~1.0）
- **最终**：非负归一化 `1 - span×weight/SPAN_MAX`（`SPAN_MAX=√2`）
- **权重**：`grow_weight` 越大线越长（0.2→跨度0.46，1.0→0.74，3.0→0.81）

### 6.3 非负改写 + 贡献占比日志
- **问题**：`interior`/`grow` 是"负损失"（奖励取负），与正的正则项**符号相反、量纲不同**，难以比较权重
- **改法**：
  - 非负改写（梯度**完全等价**）：`interior: -mean(dist) → mean(1-dist)`；`grow: -mean(span·w) → mean(1-span·w/SPAN_MAX)`
  - 新增 `loss_shares(loss_dict)`：各项占总损失的百分比，直接看"谁在主导"
- **验证**：8 项全 ≥0、占比之和=1、梯度逐位等价

### 6.4 loss 明细接口 + 可视化
- `head.loss_breakdown()` 返回**未聚合**的 8 项：
  - per-query `(B, N)`：`cost` / `interior` / `smooth` / `length` / `order` / `grow`
  - per-pair `(B, N, N)`：`repel` / `spread`
- `infer.py`：
  - 控制台打印 per-query 表格（每条线的各损失）
  - per-pair 热力图（`repel`/`spread` 的 N×N），交互模式开**第二个窗口**，`--save` 模式存 `{token}_loss.png`

---

## 7. 验证结果

- **数值**：各聚合函数与改写前逐位等价；梯度等价（grow 误差 0.0）
- **非负/占比**：8 项 ≥0，share 之和 = 1
- **训练 smoke**：各损失稳定下降、占比日志正常
- **可视化**：四联面板（分割上色 | 代价图 JET | 距离场 VIRIDIS | 预测线叠加）正常

---

## 8. 待办 / 未解决问题

1. **长度自适应（该长则长、该短则短）** —— **已讨论、暂缓**
   - 现状：`grow` 是"全局统一"先验（所有线越长越好），但正确长度应由区域结构决定
   - 已确认场景：用户要的"应该短"主要指**被非可通行区截断（尽头）**的情形
   - 选定方案：**方案 C 骨架覆盖监督**（离线对可通行区骨架化 → 用 Chamfer/双向覆盖损失让线贴合骨架，长度由骨架分支天然决定）——**尚未实现**
2. **朝向**：当前不做矫正（2D 原生对齐）；若上游分割图朝向变化需再议
3. **部署**：推理输入若为上游模型的**预测分割**（含噪声），需增加输入噪声增广提升鲁棒性（目前未做）
4. **权重调参**：`grow_weight`、`repel_margin`、各项权重需结合训练效果微调

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
