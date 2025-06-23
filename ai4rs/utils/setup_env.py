# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in ai4rs into the registries.

    Args:
        init_default_scope (bool): Whether initialize the ai4rs default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `ai4rs`, an ai4rs all registries will build modules from ai4rs's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import ai4rs.datasets  # noqa: F401,F403
    import ai4rs.evaluation  # noqa: F401,F403
    import ai4rs.models  # noqa: F401,F403
    import ai4rs.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('ai4rs')
        if never_created:
            DefaultScope.get_instance('ai4rs', scope_name='ai4rs')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'ai4rs':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "ai4rs", '
                          '`register_all_modules` will force the current'
                          'default scope to be "ai4rs". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'ai4rs-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='ai4rs')
