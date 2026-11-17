#!/usr/bin/env bash
# enter_npu.sh — 本地一键登录服务器并进入 map_det 容器
# 用法:
#   ./scripts/enter_npu.sh                                          # 进入交互 shell
#   ./scripts/enter_npu.sh python train.py configs_npu/stage1.py    # 直接执行命令
#
# 说明: 自动 SSH 登录 -> 确保容器存在/运行 -> 进入容器, 已 source Ascend 环境并 cd /workspace/code
set -euo pipefail

# ===== 服务器 / 容器配置 (可用环境变量覆盖) =====
HOST="${NPU_HOST:-172.16.103.249}"
USER="${NPU_USER:-root}"
PASS="${NPU_PASS:-Huawei@123}"
CONTAINER="${NPU_CONTAINER:-map_det}"
IMAGE="${NPU_IMAGE:-798ed037126e}"
REMOTE_BASE="${NPU_BASE:-/data01/map_det}"
NPU_DEVICES="${NPU_DEVICES:-4,5,6,7}"
SHM="${NPU_SHM:-160g}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
          -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
ssh_run() { sshpass -p "$PASS" ssh "${SSH_OPTS[@]}" "$USER@$HOST" "$@"; }

command -v sshpass >/dev/null || { echo "缺少 sshpass: sudo apt install sshpass"; exit 1; }

echo "=== 连接服务器 $USER@$HOST ==="
ssh_run "echo CONNECTED; hostname" >/dev/null

echo "=== 检查远程工作目录 ==="
ssh_run "test -d '$REMOTE_BASE' || { echo '缺少 $REMOTE_BASE, 请先运行 scripts/upload_to_npu.sh'; exit 1; }"

echo "=== 检查容器 $CONTAINER ==="
state=$(ssh_run "docker inspect -f '{{.State.Status}}' $CONTAINER 2>/dev/null || echo missing")
case "$state" in
  running)
    echo "容器已在运行"
    ;;
  missing)
    echo "容器不存在, 创建中..."
    ssh_run "docker run -itd --name $CONTAINER --shm-size $SHM \
        -e ASCEND_VISIBLE_DEVICES=$NPU_DEVICES \
        -e TORCH_HOME=/workspace/weights/torch_home \
        -v $REMOTE_BASE:/workspace \
        $IMAGE -c 'sleep infinity'"
    ;;
  *)
    echo "容器状态=$state, 启动中..."
    ssh_run "docker start $CONTAINER"
    ;;
esac

# 等待容器进入 running
for _ in $(seq 1 15); do
  [ "$(ssh_run "docker inspect -f '{{.State.Status}}' $CONTAINER" 2>/dev/null || echo missing)" = running ] && break
  sleep 1
done
if [ "$(ssh_run "docker inspect -f '{{.State.Status}}' $CONTAINER" 2>/dev/null || echo missing)" != running ]; then
  echo "[错误] 容器 $CONTAINER 未能启动"; exit 1
fi

# 挂载路径校验 (仅告警)
src=$(ssh_run "docker inspect -f '{{range .Mounts}}{{.Source}}{{end}}' $CONTAINER")
[ "$src" = "$REMOTE_BASE" ] || echo "[警告] 容器挂载为 $src, 预期 $REMOTE_BASE"

# 构造容器内命令 (限制线程/降低日志/开启 task queue)
PRELUDE="source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; \
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GLOG_v=0 TASK_QUEUE_ENABLE=2; \
cd /workspace/code"
if (( $# > 0 )); then
  printf -v USER_CMD '%q ' "$@"
  INNER="$PRELUDE; $USER_CMD"
else
  INNER="$PRELUDE; echo '[NPU] 已进入容器 $CONTAINER, 当前目录: '\$(pwd); echo '[NPU] 例: python train.py configs_npu/stage1.py'; exec bash"
fi

echo "=== 进入容器 $CONTAINER (NPU $NPU_DEVICES) ==="
REMOTE_CMD="docker exec -it $CONTAINER bash -lc $(printf '%q' "$INNER")"
exec sshpass -p "$PASS" ssh -tt "${SSH_OPTS[@]}" "$USER@$HOST" "$REMOTE_CMD"
