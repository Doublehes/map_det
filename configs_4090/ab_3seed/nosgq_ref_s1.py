"""3-seed: non-SGQ refine, seed 1."""
from configs_4090.ab_3seed._common import make_cfg

config_default = make_cfg(False, True, 1, './work_dirs/ab_3seed/nosgq_ref_s1')
