#!/usr/bin/env bash
# upload_to_npu.sh — 上传数据集/代码/权重/离线轮子到 NPU 服务器
# 用法:
#   ./upload_to_npu.sh data      # 上传 53G 数据集 (默认)
#   ./upload_to_npu.sh code      # 上传工程代码 (排除 model_slim 等)
#   ./upload_to_npu.sh weights   # 上传 resnet50 权重 (按 torchvision 期望名放置)
#   ./upload_to_npu.sh wheels    # 上传离线 wheel 目录
#   ./upload_to_npu.sh all       # 全部
set -euo pipefail

# ===== 服务器 =====
HOST="172.16.103.249"
USER="root"
PASS="Huawei@123"
REMOTE_BASE="/data01/map_det"

# ===== 本地路径 =====
LOCAL_DATA_ROOT="/media/double/T7 Shield/LINE_DATA_SAMPLE"
LOCAL_CODE_ROOT="/home/double/Documents/wangjiang/code/map_det"
LOCAL_WEIGHTS="/home/double/Documents/repos/maptr/ckpts/resnet50-19c8e357.pth"
LOCAL_WHEELS="/home/double/Documents/wangjiang/code/map_det/npu_wheels"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
          -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
RSYNC_RSH="sshpass -p $PASS ssh ${SSH_OPTS[*]}"

ssh_run() { sshpass -p "$PASS" ssh "${SSH_OPTS[@]}" "$USER@$HOST" "$@"; }

rsync_up() {  # $1=src $2=dst(远程路径), 其余为额外 rsync 参数
  local src="$1" dst="$2"; shift 2
  echo ">>> $src  ->  $USER@$HOST:$dst"
  rsync -a --partial --info=progress2 --stats --exclude='.DS_Store' \
        "$@" -e "$RSYNC_RSH" "$src" "$USER@$HOST:$dst"
}

preflight() {
  echo "=== 检查本地源 ==="
  [ -d "$LOCAL_DATA_ROOT" ] || { echo "本地数据目录不存在: $LOCAL_DATA_ROOT"; exit 1; }
  echo "=== 检查服务器连通性 ==="
  ssh_run "echo CONNECTED; hostname"
  echo "=== 创建远程目录 ==="
  ssh_run "mkdir -p '$REMOTE_BASE/data' '$REMOTE_BASE/code' '$REMOTE_BASE/weights/torch_home/hub/checkpoints' '$REMOTE_BASE/npu_wheels' '$REMOTE_BASE/work_dirs'"
  echo "=== 远程磁盘 ==="
  ssh_run "df -h '$REMOTE_BASE'"
  local need avail
  need=$(du -sb "$LOCAL_DATA_ROOT" | awk '{print $1}')
  avail=$(ssh_run "df -B1 --output=avail '$REMOTE_BASE' | tail -1" | tr -d ' ')
  if (( avail < need * 105 / 100 )); then
    echo "警告: 远程可用 ${avail}B 小于所需 ${need}B(含5%余量)"; exit 1
  fi
}

upload_data() {
  rsync_up "$LOCAL_DATA_ROOT/trainlabel_line_data_multiview/" \
           "$REMOTE_BASE/data/trainlabel_line_data_multiview/"
  rsync_up "$LOCAL_DATA_ROOT/trainlabel_line_multiview/" \
           "$REMOTE_BASE/data/trainlabel_line_multiview/"
}

upload_code() {
  rsync_up "$LOCAL_CODE_ROOT/" "$REMOTE_BASE/code/" \
    --exclude='model_slim/' --exclude='model_slim2/' --exclude='work_dirs/' \
    --exclude='.git/' --exclude='__pycache__/' --exclude='tmp/' --exclude='.vscode/'
}

upload_weights() {
  [ -f "$LOCAL_WEIGHTS" ] || { echo "权重不存在: $LOCAL_WEIGHTS"; exit 1; }
  rsync_up "$LOCAL_WEIGHTS" \
    "$REMOTE_BASE/weights/torch_home/hub/checkpoints/resnet50-0676ba61.pth"
}

upload_wheels() {
  [ -d "$LOCAL_WHEELS" ] || { echo "wheel 目录不存在: $LOCAL_WHEELS"; exit 1; }
  rsync_up "$LOCAL_WHEELS/" "$REMOTE_BASE/npu_wheels/"
}

verify() {
  echo "=== 本地 ==="
  du -sh "$LOCAL_DATA_ROOT"/trainlabel_line_data_multiview "$LOCAL_DATA_ROOT"/trainlabel_line_multiview
  find "$LOCAL_DATA_ROOT/trainlabel_line_data_multiview" -type f | wc -l
  echo "=== 远程 ==="
  ssh_run "du -sh '$REMOTE_BASE/data'/trainlabel_line_data_multiview '$REMOTE_BASE/data'/trainlabel_line_multiview; \
           find '$REMOTE_BASE/data/trainlabel_line_data_multiview' -type f | wc -l"
}

MODE="${1:-data}"
preflight
case "$MODE" in
  data)    upload_data ;;
  code)    upload_code ;;
  weights) upload_weights ;;
  wheels)  upload_wheels ;;
  all)     upload_data; upload_code; upload_weights; upload_wheels ;;
  *) echo "未知参数: $MODE (data|code|weights|wheels|all)"; exit 1 ;;
esac
verify
echo "=== 完成 ==="
