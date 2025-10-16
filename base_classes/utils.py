import os

from typing import Dict, Tuple
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as dist
from torchinfo import summary
import yaml

from skrl.resources.preprocessors.torch import RunningStandardScaler

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "softplus": nn.Softplus,
}

NOISE_MODELS = {
    "normal": dist.Normal,
    "uniform": dist.Uniform,

}

def get_activation(name: str) -> nn.Module:
    name_lower = name.strip().lower()
    if name_lower not in ACTIVATIONS:
        raise ValueError(f"{name} is not an acceptable activation function, provide a different one.")
    cls = ACTIVATIONS[name_lower]
    if cls is nn.Softmax:
        return cls(dim=-1)
    return cls()


def init_model_weights(model:nn.Module, mean:float=0.0, std:float=0.1, seed:int=42):
    if seed is not None:
        torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "weight" in name:
                nn.init.normal_(param, mean=mean, std=std)
            elif "bias" in name:
                nn.init.normal_(param, mean=mean, std=std)


def soft_update(target:nn.Module, behavior:nn.Module, tau:float):
    for target_param, source_param in zip(target.parameters(), behavior.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)


def load_config(config_path:str, args) -> Dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config.update(vars(args))
    config['seed'] = args.seed
    config['device'] = args.device
    return config

def get_noise_model(cfg:Dict, source:str="action") -> dist:
    noise_type = cfg['agent'][f'{source}_noise_type']
    params = cfg['agent'][f'{source}_noise_params']
    return NOISE_MODELS[noise_type](**params)

def print_model_summary(model:nn.Module, input_size:Tuple[int])->None:
    summary(model, input_size=input_size)
    

def load_scaler(size:int, scaler_filepath: str):
    scaler = RunningStandardScaler(size=size)
    scaler.load_state_dict(torch.load(scaler_filepath))
    return scaler

def load_file(filepath: str) -> pd.DataFrame:
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        return pd.read_excel(filepath)
    elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
        return pd.read_pickle(filepath)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format for {filepath}. Supported formats are: .csv, .xlsx, .xls, .pkl, .pickle, .json")


def make_data_frame(data_dir:str) -> pd.DataFrame:
    csv_paths = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            file = os.path.join(data_dir, file_name)
            csv_paths.append(file)
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    training_data = pd.DataFrame()
    for csv_path in csv_paths:
        df = load_file(csv_path)
        df["seed"] = int(os.path.basename(csv_path).split("_")[-1].split(".")[0])
        training_data = pd.concat([training_data, df], ignore_index=True)
    return training_data