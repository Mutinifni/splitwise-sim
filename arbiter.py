import logging

from abc import ABC

from simulator import clock, schedule_event, cancel_event, reschedule_event


class Arbiter(ABC):
    """
    Arbiter allocates Processors to Application Allocators.
    It can be used to support application autoscaling.
    """
    def __init__(self,
                 cluster,
                 overheads):
        self.cluster = cluster
        self.overheads = overheads
        self.servers = cluster.servers
        self.applications = []
        self.allocators = {}

    def add_application(self, application):
        self.applications.append(application)
        self.allocators[application.application_id] = application.allocator

    def run(self):
        pass

    def allocate(self, processors, application):
        """
        Allocates processors to the application.
        """
        pass

    def deallocate(self, processors, application):
        """
        Deallocates processors from the application.
        """
        pass


class NoOpArbiter(Arbiter):
    """
    No-op Arbiter.
    """
    pass
