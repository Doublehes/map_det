# python eval.py configs_4090/stage2_joint_e72.py --checkpoint work_dirs/stage2_joint_e72/latest.pth
# python eval.py configs_4090/stage2_joint_e72_arg.py --checkpoint work_dirs/stage2_joint_e72_arg/latest.pth
# python eval.py configs_4090/stage2_joint_freeze_ec2_e72.py --checkpoint work_dirs/stage2_joint_freeze_ec2_e72/latest.pth
# python eval.py configs_4090/stage2_joint_freeze_ec2_e72_arg.py --checkpoint work_dirs/stage2_joint_freeze_ec2_e72_arg/latest.pth

# python eval.py configs_4090/stage2_joint_freeze_ec2_e72_d3.py --checkpoint work_dirs/stage2_joint_freeze_ec2_e72_d3/latest.pth
# python eval.py configs_4090/stage2_joint_freeze_ec2_e72_arg_d3.py --checkpoint work_dirs/stage2_joint_freeze_ec2_e72_arg_d3/latest.pth


python eval.py configs_4090/stage2_joint_freeze_ec2_e72_d3.py --checkpoint work_dirs/stage2_joint_freeze_ec2_e72_d3/latest.pth
python eval.py configs_4090/stage2_joint_freeze_ec2_e72.py --checkpoint work_dirs/stage2_joint_freeze_ec2_e72/latest.pth
python eval.py configs_4090/stage2_joint_e72.py --checkpoint work_dirs/stage2_joint_e72/latest.pth