import logging

from dataclasses import dataclass


@dataclass(kw_only=True)
class ModelArchitecture():
    name: str
    num_layers: int


@dataclass(kw_only=True)
class LLMArchitecture(ModelArchitecture):
    hidden_size: int
    num_heads: int


@dataclass(kw_only=True)
class ModelParallelism():
    """
    Captures the different parallelisms of a Model.
    """
    pipeline_parallelism: int
    tensor_parallelism: int

    @property
    def num_processors(self):
        """
        The number of GPUs required is the product of the parallelisms.
        """
        return self.pipeline_parallelism * self.tensor_parallelism


@dataclass(kw_only=True)
class ModelSize():
    """
    Captures the various sizes of a Model.
    """
    weights: int
    dtype_size: int

    @property
    def total_size(self):
        return self.weights


@dataclass(kw_only=True)
class Model():
    name: str
    architecture: ModelArchitecture
    parallelism: ModelParallelism
    size: ModelSize

    @property
    def size_per_processor(self):
        return self.size.total_size / self.parallelism.num_processors


@dataclass(kw_only=True)
class GenerativeLLM(Model):
    """
    Generative Large Language Model.
    NOTE: We currently don't capture embeddings, variable context lengths, etc.
    """
    context_size: int = 0
