# SplitwiseSim: An LLM Serving Cluster Simulator

SplitwiseSim is a discrete event simulator to evaluate model serving in LLM inference clusters.
It was built to evaluate [Splitwise](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/), a generative LLM inference serving technique that splits LLM inference phases across different machines.
SplitwiseSim can easily be extended to other applications and use cases.

## Setup

You can set up SplitwiseSim by installing its Python dependencies. We recommend that you start with a fresh Python environment.

```python
# Create and activate new Python environment
conda create -n splitwise-sim python=3.11
conda activate splitwise-sim

# Install dependencies
pip install -r requirements.txt
```

**NOTE**: SplitwiseSim has only been tested with Python 3.11. However, it may also work with other Python versions.

## Inputs and Outputs

SplitwiseSim takes in a configuration file [`config.yaml`](configs/config.yaml) as input, which includes the following key components:

- [Cluster configuration](configs/cluster/) that specifies the SKUs allocated in the cluster.
- [Application configuration](configs/applications/) that specifies the model, scheduler, etc. used by each LLM application. Note that we currently only support one application.
- [Request trace](#request-traces) that specifies the set of requests that arrive into the cluster.
- [Model repository](configs/model_repo/) that captures the set of models running in the cluster.
- [Hardware SKUs repository](configs/hardware_repository/) that captures the available SKUs that can be provisioned in the cluster.
- [Performance model](#performance-model) that helps estimate request runtimes different batch, model, and hardware configurations.
- [Starting state](configs/start_state/) for the cluster.

Several other components can be configured; please see `config.yaml` for details.
Note that SplitwiseSim uses [Hydra](https://hydra.cc/) for configuration management. Configurations are hierarchical, so `config.yaml` reads from other files in the `configs/` directory. You can learn more about Hydra configurations from their [documentation](https://hydra.cc/docs/intro/).

SplitwiseSim generates the following key outputs:

- Summary of request-level metrics (in `summary.csv`)
- Per-request metrics for each completed request in the trace (in `detailed/{application_id}.csv`)
- Per-request node metrics (in `request_nodes.csv`)
- Instance-level metrics (in `instances/`, with `debug` enabled)

We provide various [utility functions](notebooks/utils.py) to process outputs as shown in [`notebooks/example.ipynb`](notebooks/example.ipynb) and [`notebooks/plots.ipynb`](notebooks/plots.ipynb).

## Example Run

The simplest way to run SplitwiseSim is to execute [`run.py`](run.py), which runs with the default configuration parameters specified in [`config.yaml`](configs/config.yaml).

Below is an example script, [`scripts/run_baseline_h_example.sh`](scripts/run_baseline_h_example.sh), which overrides the default configurations to execute a `Baseline-H100` configuration with a single DGX-H100 server.

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

Specifically, each configuration setting override is supplied as a command line argument to `run.py`, and it changes a corresponding default from `config.yaml` as follows:

- Switch to the [`token_jsq`](configs/orchestrator_repo/schedulers/token_jsq.yaml) scheduler instead of the default [`round_robin`](configs/orchestrator_repo/schedulers/round_robin.yaml) scheduler, as specified in [`configs/applications/solo.yaml`](configs/applications/solo.yaml).
- Override the cluster default from [`dgx-a100`](configs/cluster/dgx-a100.yaml) to [`half_half`](configs/cluster/half_half.yaml), which has 1 DGX-A100 and 1 DGX-H100 SKU.
- Replace the number of DGX-A100 and DGX-H100 servers within the `half_half` cluster to 1 (with no effect, given the `half_half` configuration).
- Override the performance model to [`db`](configs/performance_model/db.yaml) instead of the default [`constant`](configs/performance_model/constant.yaml).
- Override the starting state from [`orca`](configs/start_state/orca.yaml) to [`baseline`](configs/start_state/baseline.yaml).
- Set the `debug` flag to `True` (changed from `False`) and `seed` to `0` (same as before, so no effect).

We can run the example scripts provided in `scripts/` directory to run a Baseline-H100 and a Splitwise-HA configuration with an example trace (`traces/test_trace.csv`).

```bash
# run simple Baseline-H100 example
./scripts/run_baseline_h_example.sh

# run simple Splitwise-HA example
./scripts/run_splitiwse_ha_example.sh
```

**NOTE**: Scripts must be run from the top-level directory.

By default, results are generated in the `results/` directory according to the output path template specified by the `output_dir` field in [`config.yaml`](configs/config.yaml).
Open [`notebooks/example.ipynb`](notebooks/example.ipynb) using Jupyter Notebook to see an example of how to easily extract the associated outputs.

## Request Traces

SplitwiseSim expects request traces in a CSV file that contains the following fields for each request:

- `request_id`: ID of the request, typically a monotonically increasing number
- `request_type`: Type of the request (e.g., DL inference, LLM inference, etc.). Use `2` for generative LLM inference (only supported type at present).
- `application_id`: ID of the application / endpoint that the request targets. Use `0` for a single application.
- `arrival_timestamp`: Timestamp at which the request arrives into the cluster.
- `batch_size`: If the request is already batched when it arrives, that can be specified here.
- `prompt_size`: Number of tokens in the input prompt of the request.
- `token_size`: Number of tokens to be generated as output by the request.

Many of these fields have limited configurability at present. A typical new trace would change the `request_id`, `arrival_timestamp`, `prompt_size`, and `token_size`. An example trace can be found in [`traces/test_trace.csv`](traces/test_trace.csv).

Splitwise was evaluated with request traces that were based off traces from production LLM inference services at Microsoft Azure.
The [`generate_trace.py`](generate_trace.py) script can automatically download the production traces and use the corresponding prompt/token size distributions to generate traces with different request rates.
It can also help generate custom traces with different kinds of distributions.
Modify and run `generate_trace.py` with desired request rates and other parameters.
By default, all traces are expected to reside in the `traces/` directory.

## Request Processing

A few high-level points to understand how SplitwiseSim processes request traces:

- All requests first arrive at a [Cluster](cluster.py)-level [Router](router.py), which forwards them to their target [Application](application.py). The Cluster also has an [Arbiter](arbiter.py) which helps reallocate [Servers](server.py) or [Processors](processor.py) between Applications. Currently, the Router and Arbiter act as no-ops, but they could be modified in the future to include smarter routing and autoscaling strategies with overheads.
- Each [Request](request.py) targets a specific [Application](application.py), which may have one or more [Instances](instance.py) that run [Models](model.py). [Applications](application.py) use [Allocators](allocator.py) for Instance spin-up/spin-down on [Processors](processor.py), and [Schedulers](scheduler.py) for routing Requests to Instances. Currently, we do not support dynamic Instance spin-up/spin-down, but rather use [start states](start_state.py) for specifying initial set of Cluster Instances.
- [Requests](request.py) are specified as a DAG of [Nodes](node.py). Request nodes may either be [Tasks](task.py) and [Flows](flow.py). Requests are processed on [Instances](instance.py) which themselves run on [Servers](server.py); specifically, Tasks are run on [Processors](processor.py), and Flows are run on [Links](interconnect.py).
- All simulation times are assumed to be specified in seconds.

## Performance Model

SplitwiseSim uses a performance model to estimate the performance of LLM inference requests on instances. The performance model is built using extensive profiling data on DGX-A100 and DGX-H100 machines. The associated raw data can be found in [`data/perf-model.csv`](data/perf-model.csv). The performance model in [`performance-model.py`](performance_model.py) reads this raw data to generate its own performance model. To extend Splitwise to different models/platforms, please add your profiling data to the data file and update the performance model.

The performance model provides exposes two functions to the simulator, which are used to estimate how long LLM inference queries run:

1. `get_duration(task, batch, instance, *args, **kwargs)`, to estimate the runtime of prompt and token tasks.
2. `get_iteration_duration(batch, instance, *args, **kwargs)`, to estimate the runtime of each batching iteratiol (e.g., from continuous batching).

Since LLMs are typically run with [iteration-level scheduling](https://www.usenix.org/conference/osdi22/presentation/yu), we primarily rely on `get_iteration_duration` in the [Instance](instance.py) implementation (e.g., ORCAInstance and SplitwiseInstance).

## Experiment Workflow

This section describes how to run larger-scale simulations spanning a variety of configurations.

### Parallel Simulations

SplitwiseSim can be run on multiple cores (on one or more machines) to evaluate many different configurations in parallel. Each simulation configuration is run in a single process on a single core.
SplitwiseSim uses [Ray](https://github.com/ray-project/ray) via the [Hydra Ray plugin](https://hydra.cc/docs/plugins/ray_launcher/) for parallelization. Subsequently, you need to manually collect results back into the same machine using sync scripts. Example sync scripts can be found in the `sync_scripts` folder.

To start a Ray cluster, run:

- `ray start --head` on the head node
- `ray start --address` on each of the worker nodes

If you do not want to use Ray, you may alternatively use the Hydra [joblib](https://hydra.cc/docs/plugins/joblib_launcher/) launcher, which only supports multi-core parallelization on a single machine.

Running in parallel requires the `--multirun` flag.
For example, to sweep over multiple seed values in parallel, use `python --multirun run.py seed=0,1,2,3,4,5,6,7,8,9` after starting the Ray cluster.

### Experiment Runs

The `scripts/` directory provides several scripts to run larger experiments, including parallel sweeps over different cluster configurations:

- To run a baseline configuration, run `./scripts/run_baseline_a.sh` or `./scripts/run_baseline_h.sh`.
- To run a Splitwise configuration, run the appropriate Splitwise-XX file under the scripts directory. For example, to run Splitwise-HA, run `./scripts/run_splitwise_ha.sh`.
- Various experiment configurations are specified in the `configs/experiment/` folder, which override the default `config.yaml`. For example, to run a sweep of iso-cost clusters, you can run `./scripts/run_isocost.sh` which corresponds to `configs/experiment/*_isocost.yaml` (warning: running this may spin up many configurations in parallel and overload machines with less DRAM; try smaller configurations to begin with).

### Gantt charts

SplitwiseSim can output metadata that can be visualized as Gantt charts for easier analysis/debugging. Gantt chart metadata can be enabled by configuring it appropriately. Custom tasks can be added by similarly including a start and end timestamp for the needed tasks. Please check out `notebooks/example.ipynb` for an example.

## Reference

If you use SplitwiseSim for your work, please cite the accompanying [paper](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/):

> Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Aashaka Shah, Saeed Maleki, Ricardo Bianchini. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting", in Proceedings of the International Symposium on Computer Architecture (ISCA 2024). ACM, Buenos Aires, Argentina, 2024.
