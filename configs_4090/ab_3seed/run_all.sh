#!/bin/bash
set -u
cd /home/flow/Code/MilitaryModel/map_det
export PYTHONUNBUFFERED=1

STATUS=work_dirs/ab_3seed/run_all.status
: > "$STATUS"

for seed in 0 1 2; do
  for name in nosgq_abs_s${seed} nosgq_ref_s${seed} sgq_abs_s${seed} sgq_ref_s${seed}; do
    if [ -f "work_dirs/ab_3seed/${name}/epoch_72.pth" ]; then
      echo "SKIP  $name (epoch_72 exists) $(date '+%F %T')" >> "$STATUS"
      continue
    fi
    echo "START $name $(date '+%F %T')" >> "$STATUS"
    conda run --no-capture-output -n deeptouchpro python train.py \
      "configs_4090/ab_3seed/${name}.py" \
      > "work_dirs/ab_3seed/logs/${name}.log" 2>&1
    echo "DONE  $name exit=$? $(date '+%F %T')" >> "$STATUS"
  done
done

echo "ALL DONE $(date '+%F %T')" >> "$STATUS"
