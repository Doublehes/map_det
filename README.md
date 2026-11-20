# MapTR 道路结构线检测 — 训练使用说明

多相机 BEV 道路结构线检测：ResNet50+FPN → BEVFormer → Transformer 解码器，输出车道线/边界线，并可选分割头、热力图头辅助监督。

- 训练分两阶段：Stage1 分割预训练 → Stage2 检测+分割联合训练
- 支持自动选择设备：**NPU > CUDA > CPU**（见 `configs/default.py::_select_device`）

---

## 1. 环境安装

```bash
pip install -r requirements.txt
```

> 服务器/无外网环境请参考第 3 节的离线依赖处理。

---

## 2. 本地训练 (CUDA / CPU)

### 2.1 数据准备

数据集目录结构（`data_root` 下）：

```
trainlabel_line_data_multiview/
└── convert/
    └── <scene_name>/
        ├── fisheye_side_left/*.png
        ├── fisheye_side_right/*.png
        └── ...
trainlabel_line_multiview/
├── sample_own_map_infos_merge_extra_train_len27139.pkl
└── sample_own_map_infos_merge_extra_val_len272.pkl
```

在配置文件中修改数据路径（示例：`configs/stage1_seg_pretrain.py`）：

```python
config_default.data.data_root = '/path/to/trainlabel_line_data_multiview'
config_default.data.train_ann_file = '/path/to/.../train_len27139.pkl'
config_default.data.val_ann_file = '/path/to/.../val_len272.pkl'
```

常用可调项：`data.batch_size`、`data.num_workers`、`num_epochs`、`work_dir`。

### 2.2 两阶段训练

```bash
# 方式一：一键两阶段
bash run_train.sh

# 方式二：手动执行
python train.py configs/stage1_seg_pretrain.py   # Stage1: 仅分割头
python train.py configs/stage2_joint.py          # Stage2: 加载 Stage1 权重联合训练
```

- 未设置 `work_dir` 时，自动使用 `./work_dirs/<配置文件名>`
- 每个 epoch 保存 `latest.pth`，每 `checkpoint_interval` 个 epoch 保存 `epoch_N.pth`
- 日志/TensorBoard：`<work_dir>/logs`

### 2.3 断点续训 / 加载预训练

在配置中设置：

```python
config_default.resume = './work_dirs/stage1_seg_pretrain/latest.pth'   # 恢复训练(含优化器)
config_default.pretrained = './work_dirs/stage1_seg_pretrain/latest.pth'  # 仅加载权重
config_default.freeze_backbone = True   # 配合 pretrained 使用
config_default.freeze_modules = ['backbone.']  # 按前缀冻结(不依赖 pretrained)
```

### 2.4 评测 / 推理 / 可视化

```bash
# 向量化 mAP 评测
python eval.py configs/stage2_joint.py --checkpoint ./work_dirs/stage2_joint/latest.pth

# 数据集可视化
python tools/visualize.py configs/stage2_joint.py --indices 0 1 2

# 推理 + BEV/热力图可视化
python tools/infer.py configs/stage2_joint.py \
  --checkpoint ./work_dirs/stage2_joint/latest.pth --num-samples 10

# 推理冒烟(不加载权重)
python test.py --config configs/stage2_joint.py
```

---

## 3. Ascend NPU 训练

### 3.1 服务器与容器

- 服务器连接信息见 `npu_info.txt`（本说明不保存明文密码）
- 镜像：`798ed037126e`（torch 2.1.0 + torch_npu 2.1.0.post8 + torchvision 0.16 + CANN 8.0.RC3）
- 默认 docker runtime 为 `ascend`，使用宿主机 **NPU 4-7**（容器内映射为 npu:0-3）
- 工作目录：宿主机 `/data01/map_det` → 容器 `/workspace`

### 3.2 准备与上传（在本机执行，本机需有外网）

```bash
# 1) 下载离线的 aarch64 依赖
mkdir -p npu_wheels
pip download shapely==2.0.6 --no-deps --only-binary=:all: \
  --platform manylinux2014_aarch64 --python-version 39 --implementation cp --abi cp39 -d npu_wheels
pip download pyquaternion --no-deps -d npu_wheels

# 2) 上传代码 / 权重 / wheels / 数据集
chmod +x scripts/upload_to_npu.sh
./scripts/upload_to_npu.sh all      # 也可分步: code / weights / wheels / data
```

### 3.3 创建容器并安装依赖

```bash
docker run -itd --name map_det --shm-size=160g \
  -e ASCEND_VISIBLE_DEVICES=4,5,6,7 \
  -e TORCH_HOME=/workspace/weights/torch_home \
  -v /data01/map_det:/workspace \
  798ed037126e -c "sleep infinity"
```

```bash
docker exec -it map_det bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install --no-index --find-links /workspace/npu_wheels shapely pyquaternion
```

之后可用一键脚本进入容器（自动登录服务器、确保容器运行、`source` Ascend 环境并 `cd /workspace/code`）：

```bash
./scripts/enter_npu.sh                                          # 进入交互 shell
./scripts/enter_npu.sh python train.py configs_npu/stage1.py    # 直接执行命令
```

### 3.4 冒烟测试

```bash
cd /workspace/code
python tools/npu_smoke_test.py --config configs_npu/stage1.py
```

应输出 `npu.is_available=True`，算子测试全部 `[OK]`，最后 `冒烟测试结果: PASS`。

### 3.5 启动训练

```bash
# 两阶段依次执行
python -u train.py configs_npu/stage1.py > /workspace/work_dirs/stage1.log 2>&1
python -u train.py configs_npu/stage2.py > /workspace/work_dirs/stage2.log 2>&1
```

NPU 专用配置：`configs_npu/stage1.py`、`configs_npu/stage2.py`（数据路径、离线权重、work_dir 已配置好）。

### 3.6 监控

```bash
tail -f /workspace/work_dirs/stage1.log
npu-smi info
ps -eo pid,cmd | grep [t]rain.py
```

### 3.7 常见问题

| 问题 | 处理 |
|---|---|
| 服务器无外网 | 用 `npu_wheels/` 离线安装；ResNet50 权重走 `model.img_backbone.weights` 本地路径 |
| `freeze_backbone` 不生效 | 该开关仅在 `pretrained` 非空时生效；无 pretrained 时用 `freeze_modules=['backbone.']` |
| 容器启动即退出 | 镜像 Entrypoint 为 bash，需 `-c "sleep infinity"` |
| 加载 cuda checkpoint 报错 | 已统一 `torch.load(map_location='cpu')` |
| 多卡训练 | 当前仅支持单卡；4 卡 HCCL DDP 尚未实现 |

---

## 4. 性能调优与 Profiling

### 4.1 已知瓶颈（Ascend 910B4 / CANN 8.0.RC3）
- 主要耗时是 deformable attention 的 `GridSampler2DGrad`（MIX_AIV），约占单步 40-60%
- 数据加载不是瓶颈（`data_t` 约 0.02s）
- 该 CANN 上 `grid_sample` 用 fp16 反而慢约 40x，`amp` 对采样无益（默认关闭）
- 实测 per-iter：4090 0.229s / NPU 优化前 0.808s / NPU 优化后 ~0.375s

### 4.2 已开启的优化（默认生效，数值等价）
- `bev_encoder.point_sampling_impl='vectorized'`：48 次循环 → 批量 einsum
- `bev_encoder.cross_attn_impl='vectorized'`：rebatch/scatter 按 batch 向量化
- `bev_encoder.cross_attn_cam_grouping=True`：按相机可见 query 数分组，避免 padding 到 max_len
  （实测可见数 `[581,768,730,1863,0,0]`，cross-attn 计算约省 2.8x）
- `worker_num_threads=1`：DataLoader worker 限制线程，避免多核超订阅
- NPU 容器内 `enter_npu.sh` 自动设置 `OMP_NUM_THREADS=1`、`GLOG_v=0`、`TASK_QUEUE_ENABLE=2`

### 4.3 可选开关

| 字段 | 默认 | 说明 |
|---|---|---|
| `cross_attn_len_bucket` | 0 | >0 时把长度对齐到桶（如 64），减少动态 shape |
| `amp` / `amp_dtype` | False | 混合精度；fp16 下 grid_sample 明显变慢，需先实测 |
| `npu_jit_compile` | False | 仅 NPU 图模式；需确认动态 shape 兼容 |
| `model.bev_encoder.num_sampling_points` | 8 | 降到 4 可近乎减半 grid_sample 计算（精度需评估） |
| `model.img_backbone.num_feat_levels` | 2 | 降到 1 可减少一次 cross-attn grid_sample |

### 4.4 工具

```bash
python tools/check_equiv.py --device cpu      # legacy vs 向量化数值等价性
python tools/profile_step.py --config configs_npu/stage1.py          # 单步模块分解
python tools/profile_ops.py --config configs_npu/stage1.py --out /tmp/npu_prof  # NPU 算子级
```

### 4.5 兼容性
- 所有优化默认保证 4090 行为不变，可通过 `point_sampling_impl='legacy'` / `cross_attn_impl='legacy'` 回退
- 修改模型结构类参数（sampling points / levels / BEV 分辨率）会改变精度，需单独评估

---

## 5. 命令速查

| 场景 | 命令 |
|---|---|
| 本地两阶段 | `bash run_train.sh` |
| 单阶段训练 | `python train.py <config>` |
| 断点续训 | config 设置 `resume` 后 `python train.py <config>` |
| 评测 | `python eval.py <config> --checkpoint <ckpt>` |
| 推理可视化 | `python tools/infer.py <config> --checkpoint <ckpt>` |
| 上传 NPU | `./scripts/upload_to_npu.sh all` |
| 进入 NPU 容器 | `./scripts/enter_npu.sh` |
| NPU 冒烟 | `python tools/npu_smoke_test.py --config configs_npu/stage1.py` |
| NPU 两阶段 | `python train.py configs_npu/stage1.py` → `configs_npu/stage2.py` |
| 等价性检查 | `python tools/check_equiv.py --device cpu` |
| NPU 单步分解 | `python tools/profile_step.py --config configs_npu/stage1.py` |
| NPU 算子 profiling | `python tools/profile_ops.py --config configs_npu/stage1.py` |

---

## 6. TODO / 待办

> 状态：代码已实现，默认行为保持不变；等待 A/B 实验。
> 最后更新：2026-11-18

| 子项 | 状态 | 备注 |
|---|---|---|
| 6.1 delta refinement | 已实现 | 方案 1 |
| 6.2 learnable initial ref points | 已实现 | 方案 2 |
| 6.3 组合方案与 A/B 实验 | 已实现，待实验 | 主实验 |

### 6.1 解码头迭代细化（delta refinement）

> 设计目标：在不改变默认行为的前提下，增加 MapTR/Deformable-DETR 风格的
> reference point 迭代细化，并支持共享/独立 reg head 的 A/B 对比。

#### 方案
- 采用方案 A（与 MapTR 原版一致）：
  `new_ref = sigmoid(inverse_sigmoid(ref_last) + reg_branch(output_l))`
- 默认 `predict_refine=False`，保持现有“绝对回归”行为不变。
- 默认 `different_heads=False`（各层共享 reg/cls head，与上游 MapTracker 配置一致）。
- delta 累加在 canonical（未 y_flip）归一化坐标的 logit 空间；`y_flip` 只作用于
  传给 cross-attn 的临时 ref。
- 保留 `reference_points = new_ref.clone().detach()`，配合 `aux_loss=True` 做逐层监督。
- 与 6.2 的可学习初始 ref 组合使用时，delta 从可学习参数出发；组合实验见 6.3。

#### 配置项（`configs/default.py` 的 `map_det_head`）
```python
'ref_init': 'mlp',          # 'mlp' | 'learnable'，详见 6.2
'predict_refine': False,    # False=绝对回归(现状); True=delta 细化(类 MapTR)
'different_heads': False,   # False=各层共享 cls/reg head; True=每层独立
```

#### 代码改动点
- `models/decoder.py`
  - 新增 `inverse_sigmoid(x, eps=1e-5)`（mmdet 风格 clamp 实现）。
  - `MapTransformerDecoder.__init__` 读取 `predict_refine`。
  - `forward` 的 reg 分支增加 delta 逻辑；`predict_refine=False` 路径保持逐位一致。
- `models/head.py`
  - 新增 `import copy`。
  - 按 `different_heads` 选择共享 module 或 `copy.deepcopy` 每层独立 head。
  - 更新相关注释；`init_weights` 逻辑无需改。
  - `reference_points_embed` 在 `ref_init='learnable'` 时替换为 `nn.Parameter`（详见 6.2）。
- 新增 A/B 配置（已创建，基于 `stage2_joint_freeze_ec2_e72_d3.py`）
  - `configs_4090/stage2_joint_freeze_ec2_e72_d3_learnable.py`
  - `configs_4090/stage2_joint_freeze_ec2_e72_d3_refine.py`
  - `configs_4090/stage2_joint_freeze_ec2_e72_d3_learnable_refine.py`
  - `configs_4090/stage2_joint_freeze_ec2_e72_d3_learnable_refine_dh.py`

#### 验证计划
- 恒等性：reg 输出置零时 `new_ref == ref_last`（delta 模式）。
- logit 往返：`inverse_sigmoid(x).sigmoid() ≈ x`，0/1 边界不产生 inf/nan。
- 默认路径回归：`predict_refine=False` 前向输出与改动前一致。
- d3 A/B（固定 seed / batch / lr）：

  | run | predict_refine | different_heads |
  |---|---|---|
  | baseline | False | False |
  | delta | True | False |
  | delta + indep（可选） | True | True |

  对比逐层 `reg_loss` 曲线、收敛速度、最终 mAP@0.5/1.0/1.5。
- 单层 smoke：`num_decoder_layers=1, predict_refine=True` 跑 1–2 epoch。
- A/B 期间保持 `amp=False`，避免 seg-head fp16 NaN 干扰。

#### 兼容性注意
- `predict_refine=True` 时 reg head 语义由“绝对坐标”变为“logit 偏移”；
  从 stage1 启动没问题（stage1 检测头关闭），但不要直接 resume 旧的
  “绝对回归” stage2 检测头，需丢弃 `head.*` 权重或重新训练。
- `different_heads=True` 可用旧共享 checkpoint 初始化（各层复制同一初值）；
  反向（独立 → 共享）不推荐。

### 6.2 可学习初始 reference points（learnable ref init）

> 方案 2：不做 anchor、不做曲线拟合，直接把初始 reference points 做成每个 query
> 独立的可学习参数，并替换掉 `MLP(query)`；默认关闭，保持现状。

#### 最终决策

| 项 | 决策 |
|---|---|
| 参数化 | `nn.Parameter(num_queries, num_points * 2)`，存 logits |
| 初始化 | xavier（复用 `init_weights` 的通用循环） |
| 是否替换 MLP | 替换 `reference_points_embed` |
| anchor | 不做 |
| 曲线 / 二次项 | 不做 |
| aux query | 同样替换为 `aux_reference_points` |
| 默认值 | `ref_init='mlp'`，不改变现有行为 |

#### 配置项（`configs/default.py` 的 `map_det_head`）
```python
'ref_init': 'mlp',   # 'mlp' = 现状; 'learnable' = 直接可学习 ref points
```

#### 代码改动点
- `models/head.py:87-99`：按 `ref_init` 条件创建 `reference_points` 或 `reference_points_embed`。
- `models/head.py:103-110`：aux query 同理（`aux_reference_points` / `aux_ref_points_embed`）。
- `models/head.py:158-166`：xavier 由现有通用循环覆盖，无需额外逻辑。
- `models/head.py:199-213`：forward 走 learnable 分支：

  ```python
  if self.ref_init_mode == 'learnable':
      ref_logits_main = self.reference_points.unsqueeze(0).expand(bs, -1, -1)
  else:
      ref_logits_main = self.reference_points_embed(query_main.permute(1, 0, 2))
  ref_points_main = ref_logits_main.sigmoid().reshape(
      bs, self.num_queries, self.num_points, 2)
  ```

  注意用 `reshape` 而不是 `view`（`expand` 后可能非连续）。

#### 验证
- `reference_points` 形状 `(num_queries, num_points*2)`，且 `reference_points_embed` 不存在。
- xavier 后 logits 标准差约 0.17，32 个 query 的初始 ref 互不相同。
- `ref_init='mlp'` 路径与改动前逐位一致。
- `predict_refine=True` 时，把 reg 输出置零，`new_ref ≈ ref_last`。
- 初始 `ref_0` 仍接近中心一团（无 anchor，预期内）；训练中监控 ref_0 覆盖度是否提升。

### 6.3 组合方案与 A/B 实验

#### 实验矩阵（1 层与 d3 都跑）

| run | ref_init | predict_refine | different_heads | 目的 |
|---|---|---|---|---|
| baseline | mlp | False | False | 现状 |
| learnable | learnable | False | False | 参数化解耦 |
| delta only | mlp | True | False | 无 anchor 的 delta |
| learnable + delta | learnable | True | False | 主目标 |
| learnable + delta + indep（可选） | learnable | True | True | head 独立性 |

#### 实验顺序
1. 1 层配置：`learnable` vs `mlp`。
2. 1 层配置：`learnable + delta` vs `mlp + delta`。
3. d3 配置：完整矩阵。
4. 可选：`different_heads=True` 单独 ablation。

#### 指标
- 早期 epoch 的逐层 `reg_loss` 与收敛速度。
- 最终 mAP@0.5/1.0/1.5。
- 端点误差、弯线子集指标。
- 每几个 epoch 计算一次 `ref_0` 覆盖度（GT 点到 ref polyline 的最小距离）。

固定 seed / batch / lr，一次只改一个开关。
