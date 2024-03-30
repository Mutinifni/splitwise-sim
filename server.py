import logging

from hydra.utils import instantiate

import utils
import hardware_repo

from power_model import get_server_power
from simulator import clock, schedule_event, cancel_event, reschedule_event

# used by hydra instantiate
import processor
import interconnect


class Server:
    """
    Servers are a collection of Processors that may be connected by
    local Interconnects. Servers themselves are also interconnected
    by Interconnects. Servers run Instances (partially or fully).

    Attributes:
        server_id (str): The unique server_id of the server.
        processors (list): A list of Processors.
        interconnects (list[Link]): Peers that this Server is
                                    directly connected to.
    """
    servers = {}
    # logger for all servers
    logger = None

    def __init__(self,
                 server_id,
                 name,
                 processors,
                 interconnects):
        if server_id in Server.servers:
            # NOTE: This is a hacky workaround for Hydra
            # Hydra multirun has a bug where it tries to instantiate a cluster again
            # with the same class, triggering this path. This likely happens because
            # Hydra multirun reuses the same classes across threads
            Server.servers = {}
            Server.logger = None
        self.server_id = server_id
        self.name = name
        self.processors = processors
        for proc in self.processors:
            proc.server = self
        self.interconnects = interconnects
        for intercon in self.interconnects:
            intercon.server = self
        self.cluster = None
        Server.servers[server_id] = self
        self.instances = []
        self.power = 0
        self.update_power(0)
        #self._instances = []

        # initialize server logger
        if Server.logger is None:
            self.logger = utils.file_logger("server")
            Server.logger = self.logger
            self.logger.info("time,server")
        else:
            self.logger = Server.logger

    def __str__(self):
        return f"Server:{self.server_id}"

    def __repr__(self):
        return self.__str__()

    @property
    def instances(self):
        return self._instances

    @instances.setter
    def instances(self, instances):
        self._instances = instances

    def update_power(self, power):
        old_power = self.power
        self.power = get_server_power(self) + \
                        sum(processor.power for processor in self.processors)
        if self.cluster:
            self.cluster.update_power(self.power - old_power)

    def run(self):
        pass

    @classmethod
    def load(cls):
        pass

    @classmethod
    def from_config(cls, *args, server_id, **kwargs):
        sku_cfg = args[0]
        processors_cfg = sku_cfg.processors
        interconnects_cfg = sku_cfg.interconnects

        processors = []
        for processor_cfg in processors_cfg:
            for n in range(processor_cfg.count):
                processor = hardware_repo.get_processor(processor_cfg.name)
                processors.append(processor)

        # TODO: add better network topology / configuration support
        interconnects = []
        for interconnect_name in interconnects_cfg:
            intercon = hardware_repo.get_interconnect(interconnect_name)
            interconnects.append(intercon)

        return cls(server_id=server_id,
                   name=sku_cfg.name,
                   processors=processors,
                   interconnects=interconnects)


if __name__ == "__main__":
    pass
