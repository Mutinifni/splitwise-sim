import logging

from enum import IntEnum

from flow import Flow
from node import NodeState
from simulator import clock, schedule_event, cancel_event, reschedule_event
from task import Task


class ExecutorType(IntEnum):
    LocalExecutor = 0
    CentralExecutor = 1


class Executor():
    """
    Executor orchestrates Request execution on Instances and Interconnects.
    Executors can themselves run anywhere, e.g., on the Scheduler, Instance, etc.,
    with different amounts of overheads.
    They could execute multiple Tasks/Flows of the Request in parallel.

    NOTE: We don't ensure predeccessors of node are completed before submit.
    Implicitly, we assume that the Request is a tree instead of a DAG.
    Could be changed by waiting on Node predecessors.
    """
    def __init__(self,
                 request,
                 scheduler,
                 overheads):
        self.request = request
        self.scheduler = scheduler
        self.overheads = overheads
        self.submitted = []
        # to cancel any events
        self.completion_events = {}

    def successors(self, node):
        """
        Returns the successors of the specified node.
        """
        nodes = self.request.successors(node)
        return nodes

    def check_predecessors(self, node):
        """
        Checks if all predecessors of the specified node are completed.
        """
        for predecessor in self.request.predecessors(node):
            if predecessor.state != NodeState.COMPLETED:
                return False
        return True

    def submit(self, node=None):
        """
        Submits the specified node for execution.
        """
        if isinstance(node, Task):
            self.submit_task(node)
        elif isinstance(node, Flow):
            self.submit_flow(node)
        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    def submit_chain(self, chain):
        """
        Submits the specified chain of Nodes for execution.
        """
        for node in chain:
            self.submit(node)

    def submit_task(self, task, instance=None):
        """
        Submits the specified task for execution.
        If instance is not specified, uses the task's instance.
        """
        if instance is None:
            instance = task.instance
        task.executor = self
        self.submitted.append(task)
        schedule_event(self.overheads.submit_task,
                       lambda instance=instance,task=task: \
                           instance.task_arrival(task))
        # if this is the first task in the chain, submit the chain
        self.submit_chain(task.chain)

    def finish_task(self, task, instance):
        """
        Finishes the specified task.
        """
        self.submitted.remove(task)
        successor_nodes = list(self.successors(task))
        # NOTE: assumes a single leaf node
        if len(successor_nodes) == 0:
            self.finish_request()
            return
        # submit nodes for whom all predecessors have completed
        # and are not already submitted
        for node in successor_nodes:
            if node.state == NodeState.NONE and self.check_predecessors(node):
                self.submit(node)

    def submit_flow(self, flow, link=None):
        """
        Submits the specified flow for execution.
        If link is not specified, uses the flow's link.
        """
        if link is None:
            link = flow.link
        flow.executor = self
        self.submitted.append(flow)
        schedule_event(self.overheads.submit_flow,
                       lambda link=link,flow=flow: link.flow_arrival(flow))
        # if this is the first flow in the chain, submit the chain
        self.submit_chain(flow.chain)

    def finish_flow(self, flow, link):
        """
        Finishes the specified flow.
        """
        self.submitted.remove(flow)
        successor_nodes = list(self.successors(flow))
        # NOTE: assumes a single leaf node
        if len(successor_nodes) == 0:
            self.finish_request()
            return
        # submit nodes for whom all predecessors have completed
        # and are not already submitted
        for node in successor_nodes:
            if node.state == NodeState.NONE and self.check_predecessors(node):
                self.submit(node)

    def finish_request(self):
        """
        Finishes executing the entire Request.
        """
        def fin_req():
            self.scheduler.request_completion(self.request)
        schedule_event(self.overheads.finish_request, fin_req)

    def run(self):
        """
        Runs the Request by submitting the root node.
        """
        self.submit(self.request.root_node)

    @classmethod
    def create(cls, executor_type, request, scheduler, overheads):
        """
        Creates an Executor instance based on the specified type.
        """
        if executor_type == ExecutorType.CentralExecutor:
            return CentralExecutor(request, scheduler, overheads)
        if executor_type == ExecutorType.LocalExecutor:
            return LocalExecutor(request, scheduler, overheads)
        raise ValueError(f"Unsupported executor type: {executor_type}")


class CentralExecutor(Executor):
    """
    CentralExecutor coordinates with Scheduler for each Task.
    Logically, it runs within Scheduler itself.
    TODO: appropriate overheads
    """
    pass


class LocalExecutor(Executor):
    """
    LocalExecutor logically runs on Servers, alongside Instances.
    TODO: appropriate overheads
    """
    pass
