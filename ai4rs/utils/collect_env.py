# Copyright (c) ai4rs. All rights reserved.
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

import ai4rs


def collect_env():
    """Collect environment information."""
    env_info = collect_base_env()
    env_info['ai4rs'] = (
        ai4rs.__version__ + '+' + get_git_hash(digits=7))
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
