# SplitwiseSim: LLM Serving Cluster Simulator

SplitwiseSim is a discrete event simulator that helps evaluate model serving in LLM inference clusters. It was built to evaluate [Splitwise](#reference), a generative LLM inference serving technique that splits LLM inference phases across different machines. SplitwiseSim can easily be extended to other applications and use cases.

## Setup

You can set up SplitwiseSim by installing its Python dependencies. We recommend starting with a fresh Python environment.

```python
# Create and activate new Python environment
conda create -n splitwise-sim python=3.11
conda activate splitwise-sim

# Install dependencies
pip install -r requirements.txt
```

**NOTE**: SplitwiseSim has only been tested with Python 3.11. However, it will likely also work with other Python versions.

## Inputs and Outputs

SplitwiseSim takes in a hierarchical set of YAML configuration files as input, and it produces several CSV files as output. It uses [Hydra](https://hydra.cc/) for configuration management. You can learn more about configuration management from the [Hydra docs](https://hydra.cc/docs/intro/).

The top-level configuration file for SplitwiseSim is [`config.yaml`](configs/config.yaml), which points to lower-level configurations specified by other files in the `configs/` directory. Specifically, `config.yaml` captures the following key components:

- [cluster](configs/cluster/): the provisioned server SKUs in the cluster, along with their respective counts.
- [trace](#request-traces): request trace that specifies the set of requests that arrive into the cluster.
- [router](configs/router/): the cluster-level router that routes incoming requests to application-level schedulers; currently a no-op.
- [arbiter](configs/arbiter/): the cluster-level arbiter that manages compute resources between applications to support autoscaling; currently a no-op.
- [application](configs/applications/): the logical endpoint that the requests target, which specifies the model and the set of instances on which the request runs; currently, we support only one application.
- [model_repo](configs/model_repo/): the set of models (LLMs) available to run in the cluster; used for dynamic model instantiation.
- [orchestrator_repo](configs/orchestrator_repo/): the set of application resource orchestrators (i.e., schedulers and allocators) in the cluster; used for dynamic application management.
- [hardware_repo](configs/hardware_repo/): the set of available SKUs that can be provisioned in the cluster; used for dynamic server instantiation.
- [performance_model](#performance-model): an analytical model that helps estimate request runtimes with different batch, model, and hardware configurations.
- [start_state](configs/start_state/): starting state for the cluster, which helps simplify evaluation.

Several other aspects can be configured; please see [`config.yaml`](configs/config.yaml) for details.

SplitwiseSim generates the following key outputs:

- Summary of application-level metrics (`summary.csv`)
- Per-request metrics for each completed request for each application (`detailed/{application_id}.csv`)
- Request node-level metrics (`request_nodes.csv`)
- Instance-level execution metrics (in `instances/`, with `debug` enabled)

We provide various [utility functions](notebooks/utils.py) to process outputs, as shown in [`notebooks/example.ipynb`](notebooks/example.ipynb) and [`notebooks/plots.ipynb`](notebooks/plots.ipynb).

## Example Run

The simplest way to run SplitwiseSim is to execute [`run.py`](run.py), which runs with the default configuration parameters specified in [`config.yaml`](configs/config.yaml). The default configurations can be overridden by specifying appropriate command line parameters using Hydra. Below is an example script, [`scripts/run_baseline_h_example.sh`](scripts/run_baseline_h_example.sh), which overrides the default configuration to execute a simple `Baseline-H100` configuration with a single DGX-H100 server.

```bash
# scripts/run_baseline_h_example.sh

SCHEDULER=token_jsq
NUM_DGX_A100=0
NUM_DGX_H100=1
START_STATE=baseline
TRACE=test_trace

python run.py \
    applications.0.scheduler=$SCHEDULER \
    cluster=half_half \
    cluster.servers.0.count=$NUM_DGX_A100 \
    cluster.servers.1.count=$NUM_DGX_H100 \
    start_state=$START_STATE \
    performance_model=db \
    trace.filename=$TRACE \
    debug=True \
    seed=0
```

Specifically, each configuration override changes a corresponding default from `config.yaml` as follows:

- `cluster=half_half` overrides the cluster default from [`dgx-a100`](configs/cluster/dgx-a100.yaml) to [`half_half`](configs/cluster/half_half.yaml), which has 1 DGX-A100 and 1 DGX-H100 server SKU by default.
- `cluster.servers.*` replace the number of DGX-A100 and DGX-H100 servers within the [`half_half`](configs/cluster/half_half.yaml) cluster to 0 and 1, respectively.
- `applications.0.scheduler=token_jsq` switches the default [`round_robin`](configs/orchestrator_repo/schedulers/round_robin.yaml) scheduler, as specified in [`configs/applications/solo.yaml`](configs/applications/solo.yaml), to the [`token_jsq`](configs/orchestrator_repo/schedulers/token_jsq.yaml) scheduler.
- `start_state=baseline` overrides the starting state from [`orca`](configs/start_state/orca.yaml) to [`baseline`](configs/start_state/baseline.yaml).
- `performance_model=db` overrides the performance model to [`db`](configs/performance_model/db.yaml) instead of the default [`constant`](configs/performance_model/constant.yaml).
- `trace.filename=test_trace` changes the trace file name (same as default, so no effect).
- `debug=True` enables the debug flag (changed from `False`)
- `seed=0` sets the seed to `0` (same as default, so no effect).

Several of the above overrides configure objects of classes specified by the `_target_` field in the corresponding configuration files.

To simulate this simple Baseline-H100 configuration with a single DGX-H100 on [`test_trace.csv`](traces/test_trace.csv), we can simply run the bash script:

```bash
# run simple Baseline-H100 example
./scripts/run_baseline_h_example.sh
```

Similarly, we could run a simple Splitwise-HA configuration, which simulates KV-cache transfers from a DGX-H100 machine to DGX-A100 machine (see [paper](#reference) for more details):

```bash

# run simple Splitwise-HA example
./scripts/run_splitwise_ha_example.sh
```

**NOTE**: Scripts must be run from the top-level directory.

Results will be generated in the `results/` directory according to the output path template specified by the `output_dir` field in [`config.yaml`](configs/config.yaml). Open [`notebooks/example.ipynb`](notebooks/example.ipynb) using Jupyter Notebook to see an example of how to easily extract the associated outputs.

## Request Traces

SplitwiseSim expects request traces in a CSV file that contains the following fields for each request:

- `request_id`: ID of the request, typically a monotonically increasing number.
- `request_type`: Type of the request (e.g., DL inference, LLM inference, etc.). Use `2` for generative LLM inference, which is the only supported type at present.
- `application_id`: ID of the application / endpoint that the request targets. Default to `0` for a single application.
- `arrival_timestamp`: Timestamp at which the request arrives into the cluster.
- `batch_size`: If the request is already batched when it arrives, that can be specified here (currently not used).
- `prompt_size`: Number of tokens in the input prompt of the request.
- `token_size`: Number of tokens to be generated as output by the request.

Many of these fields have limited configurability at present. A typical new trace would change the `request_id`, `arrival_timestamp`, `prompt_size`, and `token_size`. An example trace can be found in [`traces/test_trace.csv`](traces/test_trace.csv).

### Production Traces and Trace Generation

[Splitwise](#reference) was evaluated with request traces that were based off [production traces](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) from LLM inference services at Microsoft Azure. The [`generate_trace.py`](generate_trace.py) script can automatically download the production traces and use the corresponding prompt/token size distributions to generate request traces with different request rates. It can also help generate custom traces with different kinds of distributions. Modify and run `generate_trace.py` with desired request rates and other parameters. By default, all generated traces are expected to reside in the `traces/` directory.

## Request Processing

SplitwiseSim processes request traces as follows:

- All requests first arrive at a [Cluster](cluster.py)-level [Router](router.py), which forwards them to their target [Application](application.py). The Cluster also has an [Arbiter](arbiter.py) which helps reallocate [Servers](server.py) or [Processors](processor.py) between Applications. Currently, the Router and Arbiter act as no-ops, but they could be modified in the future to include smarter routing and autoscaling strategies with overheads.
- Each [Request](request.py) targets a specific [Application](application.py), which may have one or more [Instances](instance.py) that run [Models](model.py). [Applications](application.py) use [Allocators](allocator.py) to spin-up/spin-down Instances on [Processors](processor.py), and they use [Schedulers](scheduler.py) to load balance Requests across Instances. Currently, we do not support dynamic Instance spin-up/spin-down, but rather use [start states](start_state.py) for specifying the initial set of Cluster Instances.
- [Requests](request.py) are specified as a Directed Acyclic Graph (DAG) of [Nodes](node.py) for flexibility. Request nodes may either be [Tasks](task.py) and [Flows](flow.py). Requests are processed on [Instances](instance.py), which run on [Servers](server.py); specifically, Tasks are run on [Processors](processor.py) and Flows are run on [Links](interconnect.py).

Note that all simulation times are assumed to be specified in seconds.

## Performance Model

The [performance_model](performance_model.py) helps SplitwiseSim estimate how long requests run on diverse input, output, hardware, batch, etc. configurations. `performance_model.PerformanceModel` is an interface class which exposes the following two estimation functions to the simulator:

1. `get_duration()`: used to estimate the runtime of prompt and token tasks.
2. `get_iteration_duration()`: used to estimate the runtime of each batching iteration (e.g., from continuous batching).

Since modern LLM serving typically uses [iteration-level scheduling](https://www.usenix.org/conference/osdi22/presentation/yu), we primarily rely on `get_iteration_duration` in the [Instance](instance.py) implementation (e.g., ORCAInstance and SplitwiseInstance).

Currently, SplitwiseSim provides two concrete performance models:

1. `performance_model=constant`: This model assumes that all prompt and token tasks take a constant duration. While unrealistic, it is helpful for testing / debugging purposes.
2. `performance_model=db`: This model uses extensive profiling data from the DGX-A100 and DGX-H100 machines and is the preferable model to use for realistic simulations. The associated raw data can be found in [`data/perf-model.csv`](data/perf-model.csv). The `performance_model.DatabasePerformanceModel` class reads this raw data to build a simple linear predictor, which serves as the performance model. To extend SplitwiseSim to different LLMs/platforms, please add your profiling data to the data file and potentially update the performance model predictor.

## Experiments Workflow

This section describes how to run larger-scale simulations spanning a variety of configurations.

### Parallel Simulations

SplitwiseSim can be run on multiple cores (on one or more machines) to evaluate many different configurations in parallel. Each simulation configuration is run in a single process on a single core. SplitwiseSim uses [Ray](https://github.com/ray-project/ray) via the [Hydra Ray plugin](https://hydra.cc/docs/plugins/ray_launcher/) for parallelization.

To start a Ray cluster, run:

- `ray start --head` on the head machine.
- `ray start --address=xxx` on each of the worker machines.

See [Ray docs](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html) for more details.

If you do not want to use Ray, you may alternatively use the Hydra [joblib](https://hydra.cc/docs/plugins/joblib_launcher/) launcher, which only supports multicore parallelization on a single machine.

Running a Hydra configuration in parallel requires the `--multirun` flag. For example, to sweep over multiple seed values in parallel, use `python --multirun run.py seed=0,1,2,3,4,5,6,7,8,9` after starting the Ray cluster.

Output from multi-machine runs is stored on different machines corresponding to where each simulation configuration runs. Subsequently, you may need to manually collect results back into the same machine using sync scripts. Example sync scripts can be found in the `sync_scripts` folder.

### Experiment Runs

The `scripts/` directory provides several scripts to run larger experiments, including parallel sweeps over different cluster configurations:

- To run a baseline configuration, run `./scripts/run_baseline_a.sh` (Baseline-A100) or `./scripts/run_baseline_h.sh` (Baseline-H100).
- To run a Splitwise configuration, run the appropriate Splitwise-XX file under the scripts directory. For example, to run Splitwise-HA, run `./scripts/run_splitwise_ha.sh`.
- Various experiment configurations used in the [Splitwise paper](#reference) are specified in the `configs/experiment/` folder. For example, to run a sweep of iso-cost clusters, you can run `./scripts/run_isocost.sh` which corresponds to `configs/experiment/*_isocost.yaml` with the appropriate sweep parameters (warning: running this may spin up many configurations in parallel and take a long time; try smaller configurations to begin with).

### Experiment Plots and Gantt Charts

Outputs from experiment sweeps can be visualized by using the plotting scripts provided in `notebooks/plots.ipynb`. These scripts were used to plot some of the graphs in the [Splitwise paper](#reference).

If the `debug` flag is enabled, SplitwiseSim additionally outputs iteration-level metadata per instance (including start/end timestamps), which can be visualized as Gantt charts for analysis and debugging. Check out `notebooks/example.ipynb` for a simple example. Custom markers can be added by modifying the simulator.

## Reference

If you use SplitwiseSim in your work, please cite the accompanying [paper](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/):

> Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting", in Proceedings of the International Symposium on Computer Architecture (ISCA 2024). ACM, Buenos Aires, Argentina, 2024.
