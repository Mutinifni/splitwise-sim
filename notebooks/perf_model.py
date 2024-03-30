import logging

import pandas as pd

from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_percentage_error


class PerfModel:
    """
    Performance model independent of the simulator.
    TODO: reuse code from simulator.
    """
    def __init__(self, db_path, init=False):
        self.db = pd.read_csv(db_path,
                              dtype={"model": "category", "hardware": "category"})

        # ensure the database has the correct columns
        # and remove extraneous columns
        self.db = self.db[["model",
                           "hardware",
                           "tensor_parallel",
                           "prompt_size",
                           "batch_size",
                           "token_size",
                           "prompt_time",
                           "token_time"]]

        # convert to seconds
        self.db["prompt_time"] = self.db["prompt_time"] / 1000
        self.db["token_time"] = self.db["token_time"] / 1000

        if init:
            self.init_predictor_numtokens()

    def init_predictor_numtokens(self):
        """
        Predict using number of tokens in the batch.
        """
        self.prompt_time_predictors = {}
        self.token_time_predictors = {}
        self.prompt_time_cache = {}
        self.token_time_cache = {}

        for model in self.db["model"].unique():
            for hardware in self.db["hardware"].unique():
                for tensor_parallel in self.db["tensor_parallel"].unique():
                    mask = (self.db["model"] == model) & (self.db["hardware"] == hardware) & (self.db["tensor_parallel"] == tensor_parallel)
                    db_subset = self.db[mask].copy()
                    if len(db_subset) == 0:
                        continue
                    db_subset["batch_tokens"] = db_subset["prompt_size"] * db_subset["batch_size"]
                    x = db_subset[["batch_tokens", "prompt_time"]].groupby("batch_tokens").median().index
                    y = db_subset[["batch_tokens", "prompt_time"]].groupby("batch_tokens").median()["prompt_time"]
                    self.prompt_time_predictors[(model, hardware, tensor_parallel)] = interp1d(
                                                                    x, y, fill_value="extrapolate")
                    x = db_subset[["batch_tokens", "token_time"]].groupby("batch_tokens").median().index
                    y = db_subset[["batch_tokens", "token_time"]].groupby("batch_tokens").median()["token_time"]
                    self.token_time_predictors[(model, hardware, tensor_parallel)] = interp1d(
                                                                    x, y, fill_value="extrapolate")

    def get_prompt_time(self, model, hardware, tensor_parallel, batch_tokens):
        prompt_time = self.prompt_time_cache.get((model, hardware, tensor_parallel, batch_tokens), None)
        if prompt_time is None:
            prompt_time = float(self.prompt_time_predictors[(model, hardware, tensor_parallel)](batch_tokens))
            self.prompt_time_cache[(model, hardware, tensor_parallel, batch_tokens)] = float(prompt_time)
        return prompt_time
    
    def get_token_time(self, model, hardware, tensor_parallel, batch_tokens):
        token_time = self.token_time_cache.get((model, hardware, tensor_parallel, batch_tokens), None)
        if token_time is None:
            token_time = float(self.token_time_predictors[(model, hardware, tensor_parallel)](batch_tokens))
            self.token_time_cache[(model, hardware, tensor_parallel, batch_tokens)] = float(token_time)
        return token_time

    def add_baseline_perf(self,
                          request_df,
                          model="bloom-176b",
                          hardware="a100-80gb",
                          tensor_parallel=8):
        """
        Normalize request_df ttft and tbt wrt the model, hardware, and tensor_parallel.
        Applies the get_prompt_time and get_token_time functions.
        """
        request_df["baseline_ttft"] = request_df.apply(lambda row:
            self.get_prompt_time(model, hardware, tensor_parallel, row["prompt_sizes"]), axis=1)
        request_df["baseline_tbt"] = request_df.apply(lambda row:
            self.get_token_time(model, hardware, tensor_parallel, row["prompt_sizes"]), axis=1)
        return request_df

    @staticmethod
    def validate_model(db_path, train_test_split=0.8):
        """
        Validate the perf model.
        """
        perf_model = PerfModel(db_path, init=False)
        db = perf_model.db

        # split the data
        train_size = int(train_test_split * len(db))

        # randomize the data
        db = db.sample(frac=1)
        train_db = db.iloc[:train_size]
        test_db = db.iloc[train_size:]

        # initialize the model
        perf_model.db = train_db
        perf_model.init_predictor_numtokens()

        # validate the model
        mape = []
        for i, row in test_db.iterrows():
            prompt_time = perf_model.get_prompt_time(row["model"],
                                                     row["hardware"],
                                                     row["tensor_parallel"],
                                                     row["prompt_size"] * row["batch_size"])
            token_time = perf_model.get_token_time(row["model"],
                                                   row["hardware"],
                                                   row["tensor_parallel"],
                                                   row["prompt_size"] * row["batch_size"])
            mape.append(mean_absolute_percentage_error([row["prompt_time"], row["token_time"]],
                                                       [prompt_time, token_time]))
        return mape
