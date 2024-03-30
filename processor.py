import logging
import os

from dataclasses import dataclass, field
from enum import IntEnum

from instance import Instance
from simulator import clock, schedule_event, cancel_event, reschedule_event


class ProcessorType(IntEnum):
    DEFAULT = 0
    CPU = 1
    GPU = 2


@dataclass(kw_only=True)
class Processor():
    """
    Processor is the lowest-level processing unit that can run computations (Tasks).
    Multiple Processors constitute a Server and may be linked via Interconnects.
    For example, CPU and GPU are both different types of Processors.

    Each Processor can belong to only one Server
    Processor could eventually run multiple Instances/Tasks.

    Attributes:
        processor_type (ProcessorType): The type of the Processor.
        memory_size (float): The memory size of the Processor.
        memory_used (float): The memory used by the Processor.
        server (Server): The Server that the Processor belongs to.
        instances (list[Instance]): Instances running on this Processor.
        interconnects (list[Link]): Peers that this Processor is directly connected to.
    """
    processor_type: ProcessorType
    name: str
    server: 'Server'
    memory_size: int
    memory_used: int
    _memory_used: int = 0
    power: float = 0.
    _power: float = 0.
    instances: list[Instance] = field(default_factory=list)
    interconnects: list['Link'] = field(default_factory=list)

    @property
    def server(self):
        return self._server

    @server.setter
    def server(self, server):
        if type(server) is property:
            server = None
        self._server = server

    @property
    def memory_used(self):
        return self._memory_used

    @memory_used.setter
    def memory_used(self, memory_used):
        if type(memory_used) is property:
            memory_used = 0
        if memory_used < 0:
            raise ValueError("Memory cannot be negative")
        # if OOM, log instance details
        if memory_used > self.memory_size:
            if os.path.exists("oom.csv") is False:
                with open("oom.csv", "w", encoding="UTF-8") as f:
                    fields = ["time",
                              "instance_name",
                              "instance_id",
                              "memory_used",
                              "processor_memory",
                              "pending_queue_length"]
                    f.write(",".join(fields) + "\n")
            with open("oom.csv", "a", encoding="UTF-8") as f:
                instance = self.instances[0]
                csv_entry = []
                csv_entry.append(clock())
                csv_entry.append(instance.name)
                csv_entry.append(instance.instance_id)
                csv_entry.append(memory_used)
                csv_entry.append(self.memory_size)
                csv_entry.append(len(instance.pending_queue))
                f.write(",".join(map(str, csv_entry)) + "\n")
            # raise OOM error
            #raise ValueError("OOM")
        self._memory_used = memory_used

    @property
    def memory_free(self):
        return self.memory_size - self.memory_used

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, power):
        if type(power) is property:
            power = 0.
        if power < 0:
            raise ValueError("Power cannot be negative")
        self._power = power

    @property
    def peers(self):
        pass


@dataclass(kw_only=True)
class CPU(Processor):
    processor_type: ProcessorType = ProcessorType.CPU


@dataclass(kw_only=True)
class GPU(Processor):
    processor_type: ProcessorType = ProcessorType.GPU


if __name__ == "__main__":
    pass
