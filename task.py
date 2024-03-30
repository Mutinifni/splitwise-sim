import logging

from dataclasses import dataclass, field
from enum import IntEnum

from metrics import TaskMetrics, TaskSLO
from node import Node
from simulator import clock, schedule_event, cancel_event, reschedule_event


class TaskType(IntEnum):
    COMPUTE = 0
    PROMPT = 1
    TOKEN = 2


@dataclass(kw_only=True)
class Task(Node):
    """
    Tasks are computation nodes in the Request DAG.
    Tasks execute on Instances.

    Tasks are the computational counterparts of Flows.
    """
    task_type: TaskType
    batch_size: int = 1
    duration: float = 0.
    remaining_duration: float = 0.
    cleanup_memory: bool = True
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    slo: TaskSLO = field(default_factory=TaskSLO)
    executor: 'Executor' = None
    instances = []
    _instance = None

    def __hash__(self):
        return hash(self.node_id)

    @property
    def instance(self):
        return self._instance

    @instance.setter
    def instance(self, instance):
        if instance is self._instance:
            return
        self._instance = instance
        if instance is not None:
            self.instances.append(instance)

    @property
    def memory(self):
        return 0

    @classmethod
    def from_type(cls, task_type, **kwargs):
        if task_type == TaskType.COMPUTE:
            return ComputeTask(**kwargs)
        elif task_type == TaskType.PROMPT:
            return PromptTask(**kwargs)
        elif task_type == TaskType.TOKEN:
            return TokenTask(**kwargs)
        else:
            raise ValueError(f"Invalid TaskType {task_type}")


@dataclass(kw_only=True)
class ComputeTask(Task):
    """
    Compute tasks represent arbitrary computation.
    """
    task_type: TaskType = TaskType.COMPUTE

    def __hash__(self):
        return hash(self.node_id)

    @property
    def memory(self):
        return 0


@dataclass(kw_only=True)
class PromptTask(Task):
    """
    Prompt tasks are the prompt (prefill) computation in a generative LLM.
    They are typically the root task in a GenerativeLLMRequest.
    """
    prompt_size: int
    tokens_per_iteration: int = 0
    processing_tokens: int = 0
    processed_tokens: int = 0
    generating_tokens: int = 0
    generated_tokens: int = 0
    task_type: TaskType = TaskType.PROMPT
    cleanup_memory: bool = False

    def __post_init__(self):
        self.tokens_per_iteration = self.prompt_size

    def __hash__(self):
        return hash(self.node_id)

    @property
    def memory(self):
        num_tokens = self.prompt_size + 1
        return self.request.estimate_kv_cache_size(num_tokens=num_tokens,
                                                   model=self.instance.model)

    def max_memory(self, instance):
        num_tokens = self.prompt_size + 1
        return self.request.estimate_kv_cache_size(num_tokens=num_tokens,
                                                   model=instance.model)

    def run(self):
        super().run()

        # manage memory
        self.instance.alloc_memory(self.request, self.memory)
        self.request.memory += self.memory

    def complete_iteration(self):
        # tokens processing
        # TODO: finer-grained memory management
        self.processed_tokens += self.processing_tokens
        self.request.processed_tokens += self.processing_tokens
        self.generated_tokens += self.generating_tokens
        self.request.generated_tokens += self.generating_tokens
        self.processing_tokens = 0
        self.generating_tokens = 0

    def is_complete(self):
        return self.generated_tokens == 1

    def complete(self):
        super().complete()

        # update scheduler bookkeeping
        self.instance.sched_pending_tokens -= self.prompt_size

        # update the TTFT
        self.request.metrics.prompt_end_timestamp = clock()
        self.request.metrics.TTFT = clock() - \
                                self.request.metrics.router_arrival_timestamp

        # ensure that we processed and generated all tokens
        assert self.processed_tokens == self.prompt_size
        assert self.request.processed_tokens == self.request.prompt_size
        assert self.generated_tokens == 1

        # manage memory
        if self.cleanup_memory:
            self.instance.free_memory(self.request, self.request.memory)
            self.request.memory = 0


@dataclass(kw_only=True)
class TokenTask(Task):
    """
    Token tasks represent the token (decode) phase in a generative LLM.
    """
    token_size: int
    tokens_per_iteration: int = 1
    processing_tokens: int = 0
    processed_tokens: int = 0
    generating_tokens: int = 0
    generated_tokens: int = 0
    task_type: TaskType = TaskType.TOKEN

    def __hash__(self):
        return hash(self.node_id)

    @property
    def memory(self):
        num_tokens = self.token_size
        return self.request.estimate_kv_cache_size(num_tokens=num_tokens,
                                                   model=self.instance.model)

    def max_memory(self, instance):
        num_tokens = self.token_size
        return self.request.estimate_kv_cache_size(num_tokens=num_tokens,
                                                   model=instance.model)

    def run(self):
        super().run()

        # manage memory
        self.instance.alloc_memory(self.request, self.memory)
        self.request.memory += self.memory

    def complete_iteration(self):
        # tokens processing
        self.processed_tokens += self.processing_tokens
        self.request.processed_tokens += self.processing_tokens
        self.generated_tokens += self.generating_tokens
        self.request.generated_tokens += self.generating_tokens
        self.processing_tokens = 0
        self.generating_tokens = 0

    def is_complete(self):
        return self.generated_tokens == self.token_size

    def complete(self):
        super().complete()

        # update scheduler bookkeeping
        self.instance.sched_pending_tokens -= 1

        # ensure that we generated all tokens
        assert self.processed_tokens == self.token_size
        assert self.generated_tokens == self.token_size
        assert self.request.generated_tokens == self.request.token_size
        assert self.request.processed_tokens == self.request.prompt_size + \
                                                self.request.token_size - 1

        # manage memory
        if self.cleanup_memory:
            self.instance.free_memory(self.request, self.request.memory)
            self.request.memory = 0


if __name__ == "__main__":
    pass
