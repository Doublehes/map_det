#!/bin/bash
set -u
cd /home/flow/Code/MilitaryModel/map_det
export PYTHONUNBUFFERED=1

DATA_ROOT=/home/flow/Code/data_small/trainlabel_line_data_multiview
ROAD_ANN=/home/flow/Code/data_small/trainlabel_line_multiview/x30_jialing_road.pkl
LOOP2_ANN=/home/flow/Code/data_small/trainlabel_line_multiview/x30_jialing_20260612_loop2.pkl
STATUS=work_dirs/ab_3seed/run_eval.status
: > "$STATUS"

for seed in 0 1 2; do
  for name in nosgq_abs_s${seed} nosgq_ref_s${seed} sgq_abs_s${seed} sgq_ref_s${seed}; do
    for ep in 60 66 72; do
      CKPT="work_dirs/ab_3seed/${name}/epoch_${ep}.pth"
      if [ ! -f "$CKPT" ]; then
        echo "MISS  $name ep${ep} (no checkpoint) $(date '+%F %T')" >> "$STATUS"
        continue
      fi
      for ds in road loop2; do
        if [ "$ds" = "road" ]; then
          ANN="$ROAD_ANN"
        else
          ANN="$LOOP2_ANN"
        fi
        OUT="work_dirs/ab_3seed/${name}/eval_${ds}_ep${ep}"
        if [ -f "${OUT}/results.json" ]; then
          echo "SKIP  $name ${ds} ep${ep} (results.json exists) $(date '+%F %T')" >> "$STATUS"
          continue
        fi
        echo "START $name ${ds} ep${ep} $(date '+%F %T')" >> "$STATUS"
        conda run --no-capture-output -n deeptouchpro python eval.py \
          "configs_4090/ab_3seed/${name}.py" \
          --checkpoint "$CKPT" \
          --data_root "$DATA_ROOT" \
          --val_ann_file "$ANN" \
          --output_dir "$OUT" \
          > "work_dirs/ab_3seed/logs/eval_${name}_${ds}_ep${ep}.log" 2>&1
        echo "DONE  $name ${ds} ep${ep} exit=$? $(date '+%F %T')" >> "$STATUS"
      done
    done
  done
done

echo "ALL EVAL DONE $(date '+%F %T')" >> "$STATUS"
