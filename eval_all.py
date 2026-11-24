import os
import subprocess


if __name__ == '__main__':
    eval_exps = [
        # 'stage2_joint_freeze_ec2_e72_d3_baseline',
        # 'stage2_joint_freeze_ec2_e72_d3_learnable',
        # 'stage2_joint_freeze_ec2_e72_d3_learnable_refine',
        # 'stage2_joint_freeze_ec2_e72_d3_learnable_refine_dh',
        # 'stage2_joint_freeze_ec2_e72_d3_line',
        # 'stage2_joint_freeze_ec2_e72_d3_line_refine',
        # 'stage2_joint_freeze_ec2_e72_d3_line_refine_dh',
        # 'stage2_joint_freeze_ec2_e72_d3_refine',
        'stage2_joint_freeze_ec2_e72_d3_sgq_identity_refine'
    ]
    for exp in eval_exps:
        os.system(f'python eval.py configs_4090/{exp}.py \
            --checkpoint work_dirs/{exp}/latest.pth \
            --data_root /home/flow/Code/data_small/trainlabel_line_data_multiview \
            --val_ann_file /home/flow/Code/data_small/trainlabel_line_multiview/x30_jialing_20260612_loop2.pkl \
            --output_dir work_dirs/{exp}/eval_x30_jialing_loop2')

    for exp in eval_exps:
        os.system(f'python eval.py configs_4090/{exp}.py \
            --checkpoint work_dirs/{exp}/latest.pth \
            --data_root /home/flow/Code/data_small/trainlabel_line_data_multiview \
            --val_ann_file /home/flow/Code/data_small/trainlabel_line_multiview/x30_jialing_road.pkl \
            --output_dir work_dirs/{exp}/eval_x30_jialing_road')