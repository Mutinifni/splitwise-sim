import logging
import os
import random
import sys

import hydra

from hydra.utils import instantiate
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from simulator import TraceSimulator
from initialize import *


# register custom hydra resolver
OmegaConf.register_new_resolver("eval", eval)


def run_simulation(cfg):
    hardware_repo = init_hardware_repo(cfg)
    model_repo = init_model_repo(cfg)
    orchestrator_repo = init_orchestrator_repo(cfg)
    performance_model = init_performance_model(cfg)
    power_model = init_power_model(cfg)
    cluster = init_cluster(cfg)
    router = init_router(cfg, cluster)
    arbiter = init_arbiter(cfg, cluster)
    applications = init_applications(cfg, cluster, router, arbiter)
    trace = init_trace(cfg)
    for application in applications.values():
        router.add_application(application)
        arbiter.add_application(application)
    sim = TraceSimulator(trace=trace,
                         cluster=cluster,
                         applications=applications,
                         router=router,
                         arbiter=arbiter,
                         end_time=cfg.end_time)
    init_start_state(cfg,
                     cluster=cluster,
                     applications=applications,
                     router=router,
                     arbiter=arbiter)
    sim.run()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    # print config
    #print(OmegaConf.to_yaml(cfg, resolve=False))
    #hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    #print(OmegaConf.to_yaml(hydra_cfg, resolve=False))

    # delete existing oom.csv if any
    if os.path.exists("oom.csv"):
        os.remove("oom.csv")

    # initialize random number generator
    random.seed(cfg.seed)
    cfg.trace_dir = os.path.join(get_original_cwd(), cfg.trace_dir)

    run_simulation(cfg)


if __name__ == "__main__":
    run()
