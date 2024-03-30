import logging

from collections import defaultdict
from itertools import count

from hydra.utils import instantiate

import hardware_repo

from simulator import clock, schedule_event, cancel_event, reschedule_event
from server import Server


class Cluster:
    """
    Cluster is a collection of Servers and interconnected Links.
    """
    def __init__(self,
                 servers,
                 interconnects,
                 power_budget):
        self.servers = servers
        self.interconnects = interconnects
        self.power_budget = power_budget
        self.total_power = 0
        for sku_name in self.servers:
            for server in self.servers[sku_name]:
                server.cluster = self
                self.total_power += server.power
        self.inflight_commands = []

        # logger for simulated power usage (NOTE: currently unsupported)
        #self.power_logger = utils.file_logger("power")
        #self.power_logger.info("time,server,power")

    def __str__(self):
        return "Cluster:" + str(self.servers)

    def add_server(self, server):
        self.servers.append(server)

    def remove_server(self, server):
        self.servers.remove(server)

    def models(self):
        models = []
        for server in self.servers:
            models.extend(server.models)
        return models

    @property
    def power(self, cached=True, servers=None):
        """
        Returns the total power usage of the cluster.
        Can return the cached value for efficiency.
        TODO: unsupported
        """
        if cached and servers is None:
            return self.total_power
        if servers is None:
            servers = self.servers
        return sum(server.power() for server in servers)

    def update_power(self, power_diff):
        """
        Updates the total power usage of the cluster.
        TODO: unsupported
        """
        self.total_power += power_diff

    def power_telemetry(self, power):
        """
        Logs the power usage of the cluster.
        TODO: currently unsupported; make configurable

        Args:
            power (float): The power usage.
        """
        time_interval = 60
        schedule_event(time_interval,
                       lambda self=self, power=self.total_power: \
                           self.power_telemetry(0))

    def run(self):
        """
        Runs servers in the cluster.
        """
        # NOTE: power usage updates not supported
        power = 0
        for sku in self.servers:
            for server in self.servers[sku]:
                server.run()
                power += server.power

    @classmethod
    def from_config(cls, *args, **kwargs):
        # args processing
        cluster_cfg = args[0]
        servers_cfg = cluster_cfg.servers
        interconnects_cfg = cluster_cfg.interconnects

        # instantiate servers
        server_id = count()
        servers = defaultdict(list)
        for server_cfg in servers_cfg:
            for n in range(server_cfg.count):
                sku_cfg = hardware_repo.get_sku_config(server_cfg.sku)
                server = Server.from_config(sku_cfg, server_id=next(server_id))
                servers[server_cfg.sku].append(server)

        # instantiate interconnects
        # TODO: add better network topology / configuration support
        interconnects = []
        for interconnect_cfg in interconnects_cfg:
            if interconnect_cfg.topology == "p2p":
                continue
            interconnect = instantiate(interconnect_cfg)
            interconnects.append(interconnect)

        return cls(servers=servers,
                   interconnects=interconnects,
                   power_budget=cluster_cfg.power_budget)


if __name__ == "__main__":
    pass
