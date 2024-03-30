import logging

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class NodeMetrics():
    arrival_timestamp: float = 0.
    start_timestamp: float = 0.
    completion_timestamp: float = 0.
    run_timestamp: float = 0.
    preempt_timestamp: float = 0.
    queue_time: float = 0.
    blocked_time: float = 0.
    service_time: float = 0.
    response_time: float = 0.


@dataclass(kw_only=True)
class FlowMetrics(NodeMetrics):
    pass


@dataclass(kw_only=True)
class TaskMetrics(NodeMetrics):
    pass


@dataclass(kw_only=True)
class RequestMetrics():
    request_id: str = ''
    router_arrival_timestamp: float = 0.
    scheduler_arrival_timestamp: float = 0.
    executor_start_timestamp: float = 0.
    scheduler_completion_timestamp: float = 0.
    router_completion_timestamp: float = 0.
    router_queue_time: float = 0.
    scheduler_queue_time: float = 0.
    queue_time: float = 0.
    service_time: float = 0.
    scheduler_response_time: float = 0.
    router_response_time: float = 0.


@dataclass(kw_only=True)
class GenerativeLLMRequestMetrics(RequestMetrics):
    prompt_start_timestamp: float = 0.
    prompt_end_timestamp: float = 0.
    token_start_timestamp: float = 0.
    token_end_timestamp: float = 0.
    TTFT: float = 0.


@dataclass(kw_only=True)
class InstanceMetrics():
    spin_up_timestamp: float = 0.
    run_timestamp: float = 0.
    spin_down_timestamp: float = 0.
    busy_time: float = 0.
    interval_time: float = 0.


@dataclass(kw_only=True)
class ApplicationMetrics():
    num_requests: int = 0
    num_tasks: int = 0
    service_times: list[float] = field(default_factory=list)
    response_times: list[float] = field(default_factory=list)


@dataclass(kw_only=True)
class RouterMetrics():
    pass


@dataclass(kw_only=True)
class ArbiterMetrics():
    pass


@dataclass(kw_only=True)
class ServerMetrics():
    pass


@dataclass(kw_only=True)
class NodeSLO():
    latency: float = 0.


@dataclass(kw_only=True)
class TaskSLO(NodeSLO):
    """
    TaskSLOs capture any SLOs that are specific to a task.
    """
    pass


@dataclass(kw_only=True)
class FlowSLO(NodeSLO):
    """
    FlowSLOs capture any SLOs that are specific to a task.
    """
    pass


@dataclass(kw_only=True)
class RequestSLO():
    """
    RequestSLO captures any SLOs that are specific to a single request.
    """
    TTFT: float = float('inf')
    e2e_latency: float = float('inf')


@dataclass(kw_only=True)
class ApplicationSLO():
    """
    ApplicationSLO captures any SLOs that apply across all application requests.
    """
    TTFT: float = float('inf')
    per_token_latency: float = float('inf')
    throughput: float = 0.
