"""
Utility functions for initializing the simulation environment.
"""

import logging
import os

from hydra.utils import instantiate
from hydra.utils import get_original_cwd

from application import Application
from cluster import Cluster
from hardware_repo import HardwareRepo
from model_repo import ModelRepo
from orchestrator_repo import OrchestratorRepo
from start_state import load_start_state
from trace import Trace


def init_trace(cfg):
    trace_path = os.path.join(get_original_cwd(), cfg.trace.path)
    trace = Trace.from_csv(trace_path)
    return trace


def init_hardware_repo(cfg):
    processors_path = os.path.join(get_original_cwd(),
                                   cfg.hardware_repo.processors)
    interconnects_path = os.path.join(get_original_cwd(),
                                      cfg.hardware_repo.interconnects)
    skus_path = os.path.join(get_original_cwd(),
                             cfg.hardware_repo.skus)
    hardware_repo = HardwareRepo(processors_path,
                                 interconnects_path,
                                 skus_path)
    return hardware_repo


def init_model_repo(cfg):
    model_architectures_path = os.path.join(get_original_cwd(),
                                            cfg.model_repo.architectures)
    model_sizes_path = os.path.join(get_original_cwd(),
                                    cfg.model_repo.sizes)
    model_repo = ModelRepo(model_architectures_path, model_sizes_path)
    return model_repo


def init_orchestrator_repo(cfg):
    allocators_path = os.path.join(get_original_cwd(),
                                   cfg.orchestrator_repo.allocators)
    schedulers_path = os.path.join(get_original_cwd(),
                                   cfg.orchestrator_repo.schedulers)
    orchestrator_repo = OrchestratorRepo(allocators_path, schedulers_path)
    return orchestrator_repo


def init_performance_model(cfg):
    performance_model = instantiate(cfg.performance_model)
    return performance_model


def init_power_model(cfg):
    power_model = instantiate(cfg.power_model)
    return power_model


def init_cluster(cfg):
    cluster = Cluster.from_config(cfg.cluster)
    return cluster


def init_router(cfg, cluster):
    router = instantiate(cfg.router, cluster=cluster)
    return router


def init_arbiter(cfg, cluster):
    arbiter = instantiate(cfg.arbiter, cluster=cluster)
    return arbiter


def init_applications(cfg, cluster, router, arbiter):
    applications = {}
    for application_cfg in cfg.applications:
        application = Application.from_config(application_cfg,
                                              cluster=cluster,
                                              router=router,
                                              arbiter=arbiter)
        applications[application_cfg.application_id] = application
    return applications


def init_start_state(cfg, **kwargs):
    load_start_state(cfg.start_state, **kwargs)


if __name__ == "__main__":
    pass
