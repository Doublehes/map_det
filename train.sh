# python train.py configs_4090/stage1_seg_pretrain_e72_arg.py
# python train.py configs_4090/stage2_joint_e72_arg.py
# python train.py configs_4090/stage2_joint_freeze_e72_arg.py
# python train.py configs_4090/stage2_joint_freeze_ec2_e72_arg.py
# python train.py configs_4090/stage2_joint_freeze_ec4_e72_arg.py
# python train.py configs_4090/stage2_joint_freeze_ec2_e72_arg_d3.py

# python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3.py



# 已训练完成的基线, 避免覆盖已有 work_dirs
# python train.py configs_4090/stage2_joint_freeze_ec2_e72.py
# python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3.py
# python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3_sgq_identity.py
# python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3_sgq_mean.py
# python train.py configs_4090/stage2_joint_e72.py
# python train.py configs_4090/stage2_joint_e72_d3.py


python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3_baseline.py
python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3_learnable.py
python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3_refine.py
python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3_learnable_refine.py
python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3_learnable_refine_dh.py
python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3_line.py
python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3_line_refine.py
python train.py configs_4090/stage2_joint_freeze_ec2_e72_d3_line_refine_dh.py
