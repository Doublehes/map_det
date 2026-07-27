import os
from configs.default import AttrDict, update_config


def load_config(path: str) -> AttrDict:
    if not os.path.exists(path):
        raise FileNotFoundError(f'配置文件不存在: {path}')

    namespace = {
        'AttrDict': AttrDict,
        'update_config': update_config,
    }
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()
    exec(compile(code, path, 'exec'), namespace)

    cfg = namespace.get('config_default')
    if cfg is None:
        raise ValueError(f'配置文件中未定义 config_default 变量: {path}')
    return cfg
