"""
This file is an helper which handles all the configurations.
"""
import os
import numpy as np
import yaml
import inspect

CONFIG_FOLDER = os.path.join("..", "config")
CONFIG_COLUMNS = ["column", "desc", "dtype", "binary", "numerical", "categorical"]
BINARY_CONFIG = ["binary", "numerical", "categorical"]
DATA_FOLDER = os.path.join("..", "data")
RESULTS_FOLDER = os.path.join("..", "results")
MODELS_FOLDER = os.path.join("..", "models")
DEFAULT_SEED = 17

class Feature():

    def __init__(self, name, label, desc="No description", dtype="Missing type"):
        self.name = name
        self.label = label 
        if desc:
            self.desc = desc
        else:
            self.desc = "No description"
        if dtype:
            self.dtype = dtype
        else:
            self.dtype = "Missing type"
    
    def __repr__(self):
        return "{} ({}) - {}".format(self.label, self.dtype, self.desc)

class Config():

    features = {}
    classes = {x: [] for x in BINARY_CONFIG}

    def __init__(self, data, model, tag, seed=DEFAULT_SEED):
        self._set_model(data, model, tag, seed)
        self._set_seed()
        self._get_features()

    def _set_model(self, data, model, tag, seed):
        self.config_file = data 
        self.model_type = model
        self.config_folder = os.path.join(CONFIG_FOLDER, model)
        try:
            os.makedirs(self.config_folder)
        except:
            pass
        self.tag = tag
        self.seed = seed

    def _set_seed(self):
        np.random.seed(self.seed)

    def _get_features(self):
        import pandas as pd
        config = pd.read_csv(os.path.join(CONFIG_FOLDER, self.config_file), sep=";", decimal=".", encoding="latin1", na_filter=False)
        config = config.loc[~(config[self.model_type] == ""), CONFIG_COLUMNS + [self.model_type]].to_dict("records")
        for record in config:
            feature_name = record["column"]
            feature_label = [record[self.model_type]]
            feature_desc = record["desc"]
            feature_dtype = record["dtype"]
            if feature_label[0] in self.classes:
                self.classes[feature_label[0]].append(feature_name)
            else:
                self.classes[feature_label[0]] = [feature_name]
            for col in BINARY_CONFIG:
                if record[col]:
                    self.classes[col].append(feature_name)
                    feature_label.append(col)
            self.features[record["column"]] = Feature(name=feature_name, label=feature_label, desc=feature_desc, dtype=feature_dtype)
    
    def update(self):
        self.features = {}
        self.classes = {x: [] for x in BINARY_CONFIG}
        self._get_features()

    def save(self, name):
        with open(os.path.join(self.config_folder, "{}.save".format(name)), "w") as fp:
            yaml.dump(self.classes, fp, default_flow_style=False)

    def load_config_file(self, name):
        file_name = "{}.conf".format("_".join([self.tag, name]))
        with open(os.path.join(self.config_folder, file_name), "r") as fp:
            config = yaml.load(fp)
        return config

    def save_config_file(self, name, data):
        file_name = "{}.conf".format("_".join([self.tag, name]))
        with open(os.path.join(self.config_folder, file_name), "w") as fp:
            yaml.dump(data, fp, default_flow_style=False)

    def gen_config_from_fun(self, name, func):
        file_name = "{}.conf".format("_".join([self.tag, name]))
        args = dict(inspect.signature(func).parameters)
        data = {arg: str(args[arg]) for arg in args}
        with open(os.path.join(self.config_folder, file_name), "w") as fp:
            yaml.dump(data, fp, default_flow_style=False)