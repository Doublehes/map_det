# python eval.py configs_4090/stage2_joint_e72.py --checkpoint work_dirs/stage2_joint_e72/latest.pth
# python eval.py configs_4090/stage2_joint_e72_arg.py --checkpoint work_dirs/stage2_joint_e72_arg/latest.pth
# python eval.py configs_4090/stage2_joint_freeze_ec2_e72.py --checkpoint work_dirs/stage2_joint_freeze_ec2_e72/latest.pth
# python eval.py configs_4090/stage2_joint_freeze_ec2_e72_arg.py --checkpoint work_dirs/stage2_joint_freeze_ec2_e72_arg/latest.pth


python eval.py configs_4090/stage2_joint_freeze_ec2_e72.py \
    --checkpoint work_dirs/stage2_joint_freeze_ec2_e72/latest.pth \
    --data_root /home/flow/Code/data_small/trainlabel_line_data_multiview \
    --val_ann_file /home/flow/Code/data_small/trainlabel_line_multiview/x30_jialing_20260612_loop2.pkl \
    --output_dir work_dirs/stage2_joint_freeze_ec2_e72/eval_x30_jialing_loop2
python eval.py configs_4090/stage2_joint_freeze_ec2_e72_d3_sgq_identity.py \
    --checkpoint work_dirs/stage2_joint_freeze_ec2_e72_d3_sgq_identity/latest.pth \
    --data_root /home/flow/Code/data_small/trainlabel_line_data_multiview \
    --val_ann_file /home/flow/Code/data_small/trainlabel_line_multiview/x30_jialing_20260612_loop2.pkl \
    --output_dir work_dirs/stage2_joint_freeze_ec2_e72_d3_sgq_identity/eval_x30_jialing_loop2
python eval.py configs_4090/stage2_joint_freeze_ec2_e72_d3_sgq_mean.py \
    --checkpoint work_dirs/stage2_joint_freeze_ec2_e72_d3_sgq_mean/latest.pth \
    --data_root /home/flow/Code/data_small/trainlabel_line_data_multiview \
    --val_ann_file /home/flow/Code/data_small/trainlabel_line_multiview/x30_jialing_20260612_loop2.pkl \
    --output_dir work_dirs/stage2_joint_freeze_ec2_e72_d3_sgq_mean/eval_x30_jialing_loop2
python eval.py configs_4090/stage2_joint_e72.py \
    --checkpoint work_dirs/stage2_joint_e72/latest.pth \
    --data_root /home/flow/Code/data_small/trainlabel_line_data_multiview \
    --val_ann_file /home/flow/Code/data_small/trainlabel_line_multiview/x30_jialing_20260612_loop2.pkl \
    --output_dir work_dirs/stage2_joint_e72/eval_x30_jialing_loop2
python eval.py configs_4090/stage2_joint_e72_d3.py \
    --checkpoint work_dirs/stage2_joint_e72_d3/latest.pth \
    --data_root /home/flow/Code/data_small/trainlabel_line_data_multiview \
    --val_ann_file /home/flow/Code/data_small/trainlabel_line_multiview/x30_jialing_20260612_loop2.pkl \
    --output_dir work_dirs/stage2_joint_e72_d3/eval_x30_jialing_loop2
