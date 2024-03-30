import logging

from abc import ABC, abstractmethod

import pandas as pd

from task import PromptTask, TokenTask


power_model = None

class PowerModel(ABC):
    """
    PowerModel helps estimate power draw for Processors, Servers, and more.
    Abstract class that must be subclassed.

    TODO: unused
    """
    def __init__(self):
        global power_model
        power_model = self

    @abstractmethod
    def get_processors_power(self, task, *args, **kwargs):
        """
        Returns the power drawn by a single processor running the task.
        """
        raise NotImplementedError

    def get_server_idle_power(self, server):
        """
        Returns the idle server power.
        """
        if server.name == "dgx-a100":
            return 1500
        else:
            return 2800


class ConstantPowerModel(PowerModel):
    """
    PowerModel that returns a constant value regardless of other parameters.
    """
    def __init__(self, idle_power, prompt_power, token_power):
        super().__init__()
        self.idle_power = idle_power
        self.prompt_power = prompt_power
        self.token_power = token_power

    def get_processors_power(self, task, processors, *args, **kwargs):
        name = processors[0].name
        if task == None:
            return [self.idle_power[name]] * len(processors)
        elif isinstance(task, PromptTask):
            return [self.prompt_power[name]] * len(processors)
        elif isinstance(task, TokenTask):
            return [self.token_power[name]] * len(processors)
        else:
            raise NotImplementedError


class DatabasePowerModel(PowerModel):
    """
    PowerModel based on a CSV database of characterization runs.
    """
    def __init__(self, dbfile):
        super().__init__()
        self.db = pd.read_csv(dbfile)

    def get_power(self,
                  server,
                  model,
                  request):
        return self.db[server][model][request]


def get_processors_power(task, *args, **kwargs):
    """
    Returns the power drawn by a single processor running the task.
    """
    return power_model.get_processors_power(task, *args, **kwargs)

def get_server_power(*args, **kwargs):
    """
    Returns the idle server power.
    """
    return power_model.get_server_idle_power(*args, **kwargs)
