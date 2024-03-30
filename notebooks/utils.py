"""
Utility functions for the notebooks.
"""
import os

import numpy as np
import pandas as pd


def baseline_a100_config(num_a100,
                         start_state="baseline",
                         scheduler="token_jsq",
                         h100_cost=4.76,
                         h100_power=44,
                         a100_cost=2.21,
                         a100_power=24.8):
    config = {
        "name": f"Baseline-A100 ({num_a100}P/T)",
        "system": "Baseline-A100",
        "scheduler": f"{scheduler}",
        "start_state": start_state,
        "cluster": f"{num_a100}_0",
        "num_servers": num_a100,
        "num_a100": num_a100,
        "num_h100": 0,
        "num_prompts": num_a100,
        "num_tokens": num_a100,
        "cost": num_a100 * a100_cost,
        "power": num_a100 * a100_power,
    }
    return config

def baseline_h100_config(num_h100,
                         start_state="baseline",
                         scheduler="token_jsq",
                         h100_cost=4.76,
                         h100_power=44,
                         a100_cost=2.21,
                         a100_power=24.8):
    config = {
        "name": f"Baseline-H100 ({num_h100}P/T)",
        "system": "Baseline-H100",
        "scheduler": f"{scheduler}",
        "start_state": start_state,
        "cluster": f"0_{num_h100}",
        "num_servers": num_h100,
        "num_a100": 0,
        "num_h100": num_h100,
        "num_prompts": num_h100,
        "num_tokens": num_h100,
        "cost": num_h100 * h100_cost,
        "power": num_h100 * h100_power,
    }
    return config

def splitwise_ha_config(num_prompt,
                            num_token,
                            start_state="splitwise",
                            scheduler="mixed_pool",
                            h100_cost=4.76,
                            h100_power=44,
                            a100_cost=2.21,
                            a100_power=24.8):
    num_h100 = num_prompt
    num_a100 = num_token
    config = {
        "name": f"Splitwise-HA ({num_prompt}P, {num_token}T)",
        "system": "Splitwise-HA",
        "scheduler": f"{scheduler}",
        "start_state": f"{start_state}_1_1",
        "cluster": f"{num_token}_{num_prompt}",
        "num_servers": num_token + num_prompt,
        "num_a100": num_token,
        "num_h100": num_prompt,
        "num_prompts": num_prompt,
        "num_tokens": num_token,
        "cost": num_h100 * h100_cost + num_a100 * a100_cost,
        "power": num_h100 * h100_power + num_a100 * a100_power,
    }
    return config

def splitwise_aa_config(num_prompt,
                            num_token,
                            start_state="splitwise",
                            scheduler="mixed_pool",
                            h100_cost=4.76,
                            h100_power=44,
                            a100_cost=2.21,
                            a100_power=24.8):
    num_a100 = num_prompt + num_token
    config = {
        "name": f"Splitwise-AA ({num_prompt}P, {num_token}T)",
        "system": "Splitwise-AA",
        "scheduler": f"{scheduler}",
        "start_state": f"{start_state}_{num_prompt}_{num_token}",
        "cluster": f"{num_a100}_0",
        "num_servers": num_a100,
        "num_a100": num_a100,
        "num_h100": 0,
        "num_prompts": num_prompt,
        "num_tokens": num_token,
        "cost": num_a100 * a100_cost,
        "power": num_a100 * a100_power,
    }
    return config

def splitwise_hh_config(num_prompt,
                            num_token,
                            start_state="splitwise",
                            scheduler="mixed_pool",
                            h100_cost=4.76,
                            h100_power=44,
                            a100_cost=2.21,
                            a100_power=24.8):
    num_h100 = num_prompt + num_token
    config = {
        "name": f"Splitwise-HH ({num_prompt}P, {num_token}T)",
        "system": "Splitwise-HH",
        "scheduler": f"{scheduler}",
        "start_state": f"{start_state}_{num_prompt}_{num_token}",
        "cluster": f"0_{num_h100}",
        "num_servers": num_h100,
        "num_a100": 0,
        "num_h100": num_h100,
        "num_prompts": num_prompt,
        "num_tokens": num_token,
        "cost": num_h100 * h100_cost,
        "power": num_h100 * h100_power,
    }
    return config

def splitwise_hhcap_config(num_prompt,
                             num_token,
                             start_state="splitwisehhcap",
                             scheduler="mixed_pool",
                             h100_cost=4.76,
                             h100_power=44,
                             a100_cost=2.21,
                             a100_power=24.8,
                             power_cap_scaler=0.7):
    num_h100 = num_prompt + num_token
    config = {
        "name": f"Splitwise-HHcap ({num_prompt}P, {num_token}T)",
        "system": "Splitwise-HHcap",
        "scheduler": f"{scheduler}",
        "start_state": f"{start_state}_1_1",
        "cluster": f"{num_token}_{num_prompt}",
        "num_servers": num_h100,
        "num_a100": 0,
        "num_h100": num_h100,
        "num_prompts": num_prompt,
        "num_tokens": num_token,
        "cost": num_h100 * h100_cost,
        "power": num_prompt * h100_power + num_token * h100_power * power_cap_scaler,
    }
    return config

def get_summary_data(results_dir, scheduler, start_state, cluster, trace, seed, model=""):
    try:
        summary_df = pd.read_csv(f"{results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/summary.csv")
    except Exception as e:
        print(e)
        print(f"Failed to read {results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/summary.csv")
        return None
    return summary_df

def get_request_data(results_dir, scheduler, start_state, cluster, trace, seed, model=""):
    try:
        request_df = pd.read_csv(f"{results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/detailed/0.csv")
    except:
        print(f"Failed to read {results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/detailed/0.csv")
        return None
    return request_df

def get_request_nodes(results_dir, scheduler, start_state, cluster, trace, seed, model=""):
    try:
        request_nodes_df = pd.read_csv(f"{results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/request_nodes.csv")
        request_nodes_df["start_timestamp_dt"] = pd.to_datetime(request_nodes_df["start_timestamp"], unit="s")
        request_nodes_df["completion_timestamp_dt"] = pd.to_datetime(request_nodes_df["completion_timestamp"], unit="s")
    except:
        print(f"Failed to read {results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/request_nodes.csv")
        return None
    return request_nodes_df

def get_instances_data(results_dir, scheduler, start_state, cluster, num_servers, trace, seed, model=""):
    try:
        instance_dfs = []
        application_id = 0
        for idx in range(num_servers):
            filename = f"{results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/instances/{application_id}/{idx}.csv"
            filepath = os.path.join(results_dir, filename)
            df = pd.read_csv(filepath)
            df["iteration"] = range(len(df))
            instance_dfs.append(df)
        instances_df = pd.concat(instance_dfs)
        instances_df["iteration_start_dt"] = pd.to_datetime(instances_df["iteration_start"], unit="s")
        instances_df["iteration_end_dt"] = pd.to_datetime(instances_df["iteration_end"], unit="s")
        instances_df["duration"] = (instances_df["iteration_end"] - instances_df["iteration_start"])
        instances_df["memory"] /= 1024 * 1024 * 1024
        return instances_df
    except:
        print(f"Failed to read {results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/instances/0/*.csv")
        return None

def get_num_batch_tokens_baseline(instances_df):
    num_batch_tokens = []
    for row in instances_df.iterrows():
        num_batch_tokens.extend(int(row[1]["num_contiguous_iterations"]) * [row[1]["batch_tokens"]])
    return num_batch_tokens

def get_num_batch_tokens_splitwise(instances_df):
    num_prompt_batch_tokens = []
    num_token_batch_tokens = []
    for row in instances_df.iterrows():
        if row[1]["tag"] == "prompt":
            num_prompt_batch_tokens.extend(int(row[1]["num_contiguous_iterations"]) * [row[1]["batch_tokens"]])
        else:
            num_token_batch_tokens.extend(int(row[1]["num_contiguous_iterations"]) * [row[1]["batch_tokens"]])
    return num_prompt_batch_tokens, num_token_batch_tokens

def get_time_duration_batch_tokens(instances_df):
    instances_df = instances_df.copy()
    return instances_df.groupby("batch_tokens").sum()["duration"]

def count_token_on_prompt_servers(instances_df, request_nodes_df):
    prompt_nodes = instances_df[instances_df["tag"] == "prompt"]["name"].unique()
    count = len(request_nodes_df[(request_nodes_df["node_type"] == "TOKEN") & 
                             (request_nodes_df["runner"].isin(prompt_nodes))])
    num_requests = request_nodes_df["request_id"].nunique()
    return count, num_requests, len(prompt_nodes)

def get_summary_data_with_config(results_dir, config, trace, seed, model=""):
    scheduler = config["scheduler"]
    start_state = config["start_state"]
    cluster = config["cluster"]
    return get_summary_data(results_dir, scheduler, start_state, cluster, trace, seed, model)

def get_request_data_with_config(results_dir, config, trace, seed, model=""):
    scheduler = config["scheduler"]
    start_state = config["start_state"]
    cluster = config["cluster"]
    return get_request_data(results_dir, scheduler, start_state, cluster, trace, seed, model)

def get_request_nodes_with_config(results_dir, config, trace, seed, model=""):
    scheduler = config["scheduler"]
    start_state = config["start_state"]
    cluster = config["cluster"]
    return get_request_nodes(results_dir, scheduler, start_state, cluster, trace, seed, model)

def get_instances_data_with_config(results_dir, config, trace, seed, model=""):
    scheduler = config["scheduler"]
    start_state = config["start_state"]
    cluster = config["cluster"]
    num_servers = config["num_servers"]
    return get_instances_data(results_dir, scheduler, start_state, cluster, num_servers, trace, seed, model)

def find_within_slo(results_df, slos):
    configs_within_slo = []
    for system_name in results_df["system"].unique():
        system_df = results_df[results_df["system"] == system_name]
        for key, value in slos.items():
            system_df = system_df[system_df[f"{key}"] < value]
        configs_within_slo.append(system_df)
    return pd.concat(configs_within_slo)

def find_cheapest(results_df):
    configs = []
    for system_name in results_df["system"].unique():
        system_df = results_df[results_df["system"] == system_name]
        cheapest = system_df[system_df["cost"] == system_df["cost"].min()]
        configs.append(cheapest)
    return pd.concat(configs)

def find_least_power(results_df):
    configs = []
    for system_name in results_df["system"].unique():
        system_df = results_df[results_df["system"] == system_name]
        least_power = system_df[system_df["power"] == system_df["power"].min()]
        configs.append(least_power)
    return pd.concat(configs)

def find_least_count(results_df):
    configs = []
    for system_name in results_df["system"].unique():
        system_df = results_df[results_df["system"] == system_name]
        least_count = system_df[system_df["num_servers"] == system_df["num_servers"].min()]
        configs.append(least_count)
    return pd.concat(configs)

def find_max_throughput(results_df):
    if "throughput" not in results_df.columns:
        # add a throughput column using the trace field
        results_df["throughput"] = results_df["trace"].apply(lambda x: int(x.split("_")[2]))
    configs = []
    for system_name in results_df["system"].unique():
        system_df = results_df[results_df["system"] == system_name]
        max_throughput = system_df[system_df["throughput"] == system_df["throughput"].max()]
        configs.append(max_throughput)
    return pd.concat(configs)
