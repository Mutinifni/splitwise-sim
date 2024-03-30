import logging

from dataclasses import dataclass, field
from enum import IntEnum

from flow import Flow
from processor import CPU, GPU
from simulator import clock, schedule_event, cancel_event, reschedule_event
from server import Server


class LinkType(IntEnum):
    DEFAULT = 0
    PCIeLink = 1
    EthernetLink = 2
    IBLink = 3
    NVLink = 4
    RDMADirectLink = 5
    DummyLink = 6


@dataclass(kw_only=True)
class Link():
    """
    Links are unidirectional edges in the cluster interconnect topology graph.
    They are the lowest-level networking equivalent of Processors.
    Instead of Tasks, Links can run (potentially multiple) Flows.
    Links have a maximum bandwidth they can support, after which point they become congested.

    TODO: replace with a higher-fidelity network model (e.g., ns-3).

    Attributes:
        link_type (LinkType): Type of the Link (e.g., NVLink, IB, etc).
        src (object): Source endpoint
        dest (object): Destination endpoint
        bandwidth (float): The maximum bandwidth supported by the Link.
        bandwidth_used (float): The bandwidth used by the Link.
        server (Server): The Server that the Processor belongs to.
        flows (List[Flow]): Flows running on this Link.
        max_flows (int): Maximum number of flows that can run in parallel on the link.
    """
    link_type: LinkType = LinkType.DEFAULT
    name: str
    src: object
    dest: object
    bandwidth: float
    bandwidth_used: float
    _bandwidth_used: float = 0
    max_flows: int
    retry: bool = True
    retry_delay: float = 1.
    overheads: dict = field(default_factory=dict)

    # queues
    pending_queue: list[Flow] = field(default_factory=list)
    executing_queue: list[Flow] = field(default_factory=list)
    completed_queue: list[Flow] = field(default_factory=list)

    @property
    def bandwidth_used(self):
        return self._bandwidth_used

    @bandwidth_used.setter
    def bandwidth_used(self, bandwidth_used):
        if type(bandwidth_used) is property:
            bandwidth_used = 0
        if bandwidth_used < 0:
            raise ValueError("Bandwidth used cannot be negative")
        elif bandwidth_used > self.bandwidth:
            raise ValueError("Cannot exceed link bandwidth")
        self._bandwidth_used = bandwidth_used

    @property
    def bandwidth_free(self):
        return self.bandwidth - self.bandwidth_used

    @property
    def peers(self):
        pass

    def flow_arrival(self, flow):
        """
        Flow arrives at the Link.
        """
        flow.instance = self
        flow.arrive()
        self.pending_queue.append(flow)
        if len(self.pending_queue) > 0 and len(self.executing_queue) < self.max_flows:
            if flow.dest.memory + flow.request.memory <= flow.dest.max_memory:
                self.run_flow(flow)
            elif self.retry:
                schedule_event(self.retry_delay, lambda link=self,flow=flow: link.retry_flow(flow))
            else:
                # will lead to OOM
                self.run_flow(flow)

    def flow_completion(self, flow):
        """
        Flow completes on this Link.
        """
        flow.complete()
        self.executing_queue.remove(flow)
        self.completed_queue.append(flow)
        flow.executor.finish_flow(flow, self)
        if flow.notify:
            flow.src.notify_flow_completion(flow)
        self.bandwidth_used -= (self.bandwidth - self.bandwidth_used)
        if len(self.pending_queue) > 0 and len(self.executing_queue) < self.max_flows:
            next_flow = self.pending_queue[0]
            if next_flow.dest.memory + next_flow.request.memory <= next_flow.dest.max_memory:
                self.run_flow(next_flow)
            elif self.retry:
                schedule_event(self.retry_delay, lambda link=self,flow=flow: link.retry_flow(flow))
            else:
                # will lead to OOM
                self.run_flow(next_flow)

    def retry_flow(self, flow):
        """
        Flow is retried on this Link.
        """
        if flow not in self.pending_queue:
            return
        if (len(self.executing_queue) < self.max_flows) and (flow.dest.memory + flow.request.memory <= flow.dest.max_memory):
            self.run_flow(flow)
        elif self.retry:
            schedule_event(self.retry_delay, lambda link=self,flow=flow: link.retry_flow(flow))
        else:
            # will lead to OOM
            self.run_flow(flow)

    def get_duration(self, flow):
        """
        FIXME: this can be shorter than prompt duration
        """
        return flow.size / (self.bandwidth - self.bandwidth_used)

    def run_flow(self, flow):
        """
        Run a Flow on this Link.
        """
        flow.run()
        self.pending_queue.remove(flow)
        self.executing_queue.append(flow)
        flow.duration = self.get_duration(flow)
        # TODO: policy on how to allocate bandwidth to multiple flows
        self.bandwidth_used += (self.bandwidth - self.bandwidth_used)
        schedule_event(flow.duration,
                       lambda link=self,flow=flow: link.flow_completion(flow))

    def preempt_flow(self, flow):
        """
        Preempt a flow on this Link.
        """
        flow.preempt()
        raise NotImplementedError


@dataclass(kw_only=True)
class PCIeLink(Link):
    """
    PCIeLink is a specific type of Link between CPUs and GPUs.
    """
    link_type: LinkType = LinkType.PCIeLink
    src: CPU
    dest: GPU


@dataclass(kw_only=True)
class EthernetLink(Link):
    """
    EthernetLink is standard Ethernet between Servers.
    """
    link_type: LinkType = LinkType.EthernetLink
    src: Server
    dest: Server


@dataclass(kw_only=True)
class IBLink(Link):
    """
    IBLink is the Infiniband Link between Servers.
    """
    link_type: LinkType = LinkType.IBLink
    src: Server
    dest: Server


@dataclass(kw_only=True)
class NVLink(Link):
    """
    NVLink is a specific type of Link between GPUs.
    """
    link_type: LinkType = LinkType.NVLink
    src: GPU
    dest: GPU


@dataclass(kw_only=True)
class RDMADirectLink(Link):
    """
    RDMADirect is the Infiniband link between GPUs across/within Servers.
    """
    link_type: LinkType = LinkType.RDMADirectLink
    src: GPU
    dest: GPU


@dataclass(kw_only=True)
class DummyLink(Link):
    """
    A Link whose bandwidth is never actually used and can hold infinite flows.
    Used to simulate delay.
    """
    link_type: LinkType = LinkType.DummyLink
    src: object = None
    dest: object = None
    max_flows: float = float("inf")

    @property
    def bandwidth_used(self):
        return self._bandwidth_used

    @bandwidth_used.setter
    def bandwidth_used(self, bandwidth_used):
        return
