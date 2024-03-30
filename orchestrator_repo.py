import logging
import os

from hydra.utils import instantiate

import utils

# needed by hydra instantiate
import allocator
import scheduler


orchestrator_repo = None


class OrchestratorRepo():
    """
    Repository of all orchestrator configs (Schedulers and Allocators).
    """
    def __init__(self,
                 allocators_path,
                 schedulers_path):
        global orchestrator_repo
        orchestrator_repo = self
        self.allocator_configs = self.get_allocator_configs(allocators_path)
        self.scheduler_configs = self.get_scheduler_configs(schedulers_path)

    def get_allocator_configs(self, allocators_path):
        return utils.read_all_yaml_cfgs(allocators_path)

    def get_scheduler_configs(self, schedulers_path):
        return utils.read_all_yaml_cfgs(schedulers_path)

    def get_allocator(self, allocator_name, application, arbiter, debug, **kwargs):
        cfg = self.allocator_configs[allocator_name]
        return instantiate(cfg,
                           application=application,
                           arbiter=arbiter,
                           debug=debug)

    def get_scheduler(self, scheduler_name, application, router, debug, **kwargs):
        cfg = self.scheduler_configs[scheduler_name]
        return instantiate(cfg,
                           application=application,
                           router=router,
                           debug=debug)


get_allocator = lambda *args,**kwargs: \
            orchestrator_repo.get_allocator(*args, **kwargs)
get_scheduler = lambda *args,**kwargs: \
            orchestrator_repo.get_scheduler(*args, **kwargs)
