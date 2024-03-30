import logging

from dataclasses import dataclass, field
from enum import IntEnum

from metrics import NodeMetrics
from simulator import clock, schedule_event, cancel_event, reschedule_event


class NodeState(IntEnum):
    NONE = 0
    QUEUED = 1
    RUNNING = 2
    BLOCKED = 3
    COMPLETED = 4
    ABORTED = 5


@dataclass(kw_only=True)
class Node():
    """
    Base class for Tasks and Nodes in a Request
    Simplest element of the Request DAG
    """
    node_id: int
    num_preemptions: int = 0
    request: 'Request' = None
    state: NodeState = NodeState.NONE
    metrics: NodeMetrics = field(default_factory=NodeMetrics)
    # chain of nodes that must be executed back-to-back
    # only stored in the first node of the chain
    chain: list = field(default_factory=list)

    def __hash__(self):
        """
        NOTE: hash functions get overridden to None in child classes
        """
        return hash(self.node_id)

    def __eq__(self, other):
        return self.node_id == other.node_id

    def arrive(self):
        assert self.state == NodeState.NONE
        self.metrics.arrival_timestamp = clock()
        self.state = NodeState.QUEUED

    def run(self):
        assert self.state == NodeState.QUEUED
        self.metrics.run_timestamp = clock()
        self.metrics.start_timestamp = clock()
        self.metrics.queue_time += clock() - self.metrics.arrival_timestamp
        if self.request.root_node is self:
            self.request.metrics.prompt_start_timestamp = clock()
            self.request.metrics.queue_time = clock() - \
                            self.request.metrics.router_arrival_timestamp
        self.state = NodeState.RUNNING

    def run_after_preempt(self):
        assert self.state == NodeState.BLOCKED
        self.metrics.run_timestamp = clock()
        self.metrics.blocked_time += clock() - self.metrics.preempt_timestamp
        self.state = NodeState.RUNNING

    def complete(self):
        assert self.state == NodeState.RUNNING
        self.metrics.completion_timestamp = clock()
        self.metrics.service_time += clock() - self.metrics.run_timestamp
        self.metrics.response_time = clock() - self.metrics.arrival_timestamp
        self.state = NodeState.COMPLETED

    def preempt(self):
        assert self.state == NodeState.RUNNING
        self.metrics.preempt_timestamp = clock()
        self.metrics.service_time += clock() - self.metrics.run_timestamp
        self.state = NodeState.BLOCKED

    def abort(self):
        if self.state == NodeState.QUEUED:
            self.metrics.queue_time += clock() - self.metrics.arrival_timestamp
            if self.request.root_node is self:
                self.request.metrics.queue_time = clock() - \
                                self.request.metrics.router_arrival_timestamp
        elif self.state == NodeState.RUNNING:
            self.metrics.service_time += clock() - self.metrics.run_timestamp
        elif self.state == NodeState.BLOCKED:
            self.metrics.blocked_time += clock() - self.metrics.preempt_timestamp
        self.state = NodeState.ABORTED
