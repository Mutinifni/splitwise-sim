import logging
import os

from hydra.utils import instantiate

import utils

# needed by hydra instantiate
import processor
import interconnect


hardware_repo = None


class HardwareRepo():
    """
    Repository of all hardware configs for dynamic instantiation.
    """
    def __init__(self,
                 processors_path,
                 interconnects_path,
                 skus_path):
        global hardware_repo
        hardware_repo = self
        self.processor_configs = self.get_processor_configs(processors_path)
        self.sku_configs = self.get_sku_configs(skus_path)
        self.interconnect_configs = self.get_interconnect_configs(
                                            interconnects_path)

    def get_sku_configs(self, skus_path):
        return utils.read_all_yaml_cfgs(skus_path)

    def get_processor_configs(self, processors_path):
        return utils.read_all_yaml_cfgs(processors_path)

    def get_interconnect_configs(self, interconnects_path):
        return utils.read_all_yaml_cfgs(interconnects_path)

    def get_processor(self, processor_name):
        cfg = self.processor_configs[processor_name]
        return instantiate(cfg)

    def get_interconnect(self, interconnect_name):
        cfg = self.interconnect_configs[interconnect_name]
        return instantiate(cfg)

    def get_sku_config(self, sku_name):
        return self.sku_configs[sku_name]


get_processor = lambda *args,**kwargs: \
                        hardware_repo.get_processor(*args, **kwargs)
get_interconnect = lambda *args,**kwargs: \
                        hardware_repo.get_interconnect(*args, **kwargs)
get_sku_config = lambda *args,**kwargs: \
                        hardware_repo.get_sku_config(*args, **kwargs)
