import logging

import utils

from hydra.utils import instantiate

from model import Model, GenerativeLLM

# needed by hydra instantiate
import model

model_repo = None


class ModelRepo():
    """
    Repository of model configurations for dynamic instantiation.
    """
    def __init__(self,
                 model_architectures_path,
                 model_sizes_path):
        global model_repo
        model_repo = self
        self.model_architecture_configs = self.get_model_architecture_configs(
                                        model_architectures_path)
        self.model_size_configs = self.get_model_size_configs(model_sizes_path)

    def get_model_architecture_configs(self, model_architectures_path):
        return utils.read_all_yaml_cfgs(model_architectures_path)

    def get_model_size_configs(self, model_sizes_path):
        return utils.read_all_yaml_cfgs(model_sizes_path)

    def get_model_architecture(self, model_architecture_name):
        cfg = self.model_architecture_configs[model_architecture_name]
        return instantiate(cfg)

    def get_model_size(self, model_size_name):
        cfg = self.model_size_configs[model_size_name]
        return instantiate(cfg)

    def get_model(self, model_architecture, model_size, model_parallelism):
        return GenerativeLLM(name=model_architecture.name,
                             architecture=model_architecture,
                             size=model_size,
                             parallelism=model_parallelism)


get_model_architecture = lambda *args,**kwargs: \
            model_repo.get_model_architecture(*args, **kwargs)
get_model_size = lambda *args,**kwargs: \
                        model_repo.get_model_size(*args, **kwargs)
get_model = lambda *args,**kwargs: \
            model_repo.get_model(*args, **kwargs)
