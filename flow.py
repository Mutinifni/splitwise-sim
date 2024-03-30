import logging

from dataclasses import dataclass, field
from enum import IntEnum

from instance import Instance
from metrics import FlowMetrics, FlowSLO
from model import Model, ModelArchitecture
from node import Node
from simulator import clock, schedule_event, cancel_event, reschedule_event


class FlowType(IntEnum):
    DEFAULT = 0
    KVCacheTransfer = 1


@dataclass(kw_only=True)
class Flow(Node):
    """
    Flows are communication nodes in the Request DAG that execute on Links.
    Flows are the networking counterparts of Tasks.
    """
    flow_type: FlowType
    src: Instance
    dest: Instance
    batch_size: int = 1
    size: float = 0.
    duration: float = 0.
    notify: bool = False
    metrics: FlowMetrics = field(default_factory=FlowMetrics)
    slo: FlowSLO = field(default_factory=FlowSLO)
    executor: 'Executor' = None
    links = []
    _link = None

    def __hash__(self):
        return hash(self.node_id)

    @property
    def link(self):
        return self._link

    @link.setter
    def link(self, link):
        if link is self._link:
            return
        self._link = link
        if link is not None:
            self.links.append(link)

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration

    @property
    def memory(self):
        return 0

    def run(self):
        super().run()

        # manage memory
        self.dest.alloc_memory(self.request, self.request.memory)

    def complete(self):
        super().complete()

        # manage memory
        self.src.free_memory(self.request, self.request.memory)

    @classmethod
    def from_type(cls, flow_type, **kwargs):
        if flow_type == FlowType.DEFAULT:
            return Flow(**kwargs)
        elif flow_type == FlowType.KVCacheTransfer:
            return KVCacheTransferFlow(**kwargs)
        else:
            raise ValueError(f"Invalid FlowType {flow_type}")


@dataclass(kw_only=True)
class KVCacheTransferFlow(Flow):
    """
    Flow for transferring KV cache between instances.
    """
    flow_type: FlowType = FlowType.KVCacheTransfer

    def __hash__(self):
        return hash(self.node_id)
