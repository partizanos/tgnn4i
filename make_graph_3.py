import torch
import torch_geometric as ptg
import numpy as np
import argparse
import os
import pandas as pd
import utils
from tqdm import tqdm
import logging
import pdb

# Configuration dictionary
config = {
    "data_dir": "step_2_mimic_demo/",
    "output_dir": "dataset/g3_mimic_demo/",
    "max_time_steps": 200,
    "max_features": 50,
    "seed": 42,
    "time_steps_upper_limit": 400,
    "batch_size": 1,
    "graph_edges": 50
}

# Function to process split data and pad tensors
def process_split_data(split_data, split):
    max_shape = (config["max_time_steps"], 1, config["max_features"], 1)
    split_data[split]['y'] = [pad_tensor(y.unsqueeze(-1), max_shape) for y in split_data[split]['y']]
    assert all(y.ndim == 4 for y in split_data[split]['y']), "y tensor must have 4 dimensions"
    
    split_data[split]['t'] = [pad_tensor(t.reshape(-1, 1, 1), (config["max_time_steps"], 1, 1)) for t in split_data[split]['t']]
    assert all(t.ndim == 3 for t in split_data[split]['t']), "t tensor must have 3 dimensions"
    
    split_data[split]['delta_t'] = [pad_tensor(delta_t.unsqueeze(-1), max_shape) for delta_t in split_data[split]['delta_t']]
    assert all(delta_t.ndim == 4 for delta_t in split_data[split]['delta_t']), "delta_t tensor must have 4 dimensions"
    
    split_data[split]['mask'] = [pad_tensor(mask.unsqueeze(-1), max_shape) for mask in split_data[split]['mask']]
    assert all(mask.ndim == 4 for mask in split_data[split]['mask']), "mask tensor must have 4 dimensions"

def align_times(t_list):
    common_times = torch.unique(torch.cat(t_list))
    if config["max_time_steps"] == None:
        max_time_steps = min(config["time_steps_upper_limit"], len(common_times))
        config["max_time_steps"] = max_time_steps
    # Adjust common_times to max_time_steps
    if len(common_times) > config["max_time_steps"]:
        common_times = common_times[:config["max_time_steps"]]
    elif len(common_times) < config["max_time_steps"]:
        pad_size = config["max_time_steps"] - len(common_times)
        common_times = torch.cat([common_times, common_times[-1:].expand(pad_size)])
    
    t_tensor = common_times.unsqueeze(0).expand(len(t_list), -1)  # Expand along the first dimension
    
    return common_times, t_tensor


# Function to initialize split data dictionary
def initialize_split_data():
    return {'train': {'y': [], 't': [], 'delta_t': [], 'mask': []},
            'val': {'y': [], 't': [], 'delta_t': [], 'mask': []},
            'test': {'y': [], 't': [], 'delta_t': [], 'mask': []}}

# Function to append data to split data dictionary
def append_split_data(split_data, split, y_tensor, t_tensor, delta_t_tensor, mask_tensor):
    split_data[split]['y'].append(y_tensor)
    split_data[split]['t'].append(t_tensor)
    split_data[split]['delta_t'].append(delta_t_tensor)
    split_data[split]['mask'].append(mask_tensor)




# Function to create a fully connected graph
def create_fully_connected_graph(num_nodes):
    return torch.randint(0, num_nodes, (2, config["graph_edges"]))

def pad_tensor(tensor, max_shape):
    assert tensor.ndim <= len(max_shape), "Tensor has more dimensions than max_shape."
    
    tensor_shape = list(tensor.shape)
    
    # Ensure all elements in max_shape are integers and defined
    max_shape = tuple(int(dim) if dim is not None else tensor_shape[i] for i, dim in enumerate(max_shape))
    
    while len(tensor_shape) < len(max_shape):
        tensor_shape.append(1)
    
    padded_tensor = torch.zeros(max_shape)
    
    # Adjust the slices to match the tensor's dimensions
    slices = tuple(slice(0, dim) for dim in tensor_shape)
    
    # Add an extra dimension if needed
    if tensor.ndim < len(max_shape):
        tensor = tensor.unsqueeze(-1)
    
    padded_tensor[slices] = tensor
    return padded_tensor




# Function to process a single stay and extract features
def process_stay(stay_id):
    set_default_config_values()
    y_list, t_list = extract_features(stay_id)
    common_times, t_tensor = align_times(t_list)
    y_tensor, mask_tensor = align_y_and_masks(y_list, t_tensor, common_times)
    delta_t_tensor = calculate_delta_t(t_tensor, y_tensor)
    return create_fully_connected_graph(len(y_list)), y_tensor, t_tensor, delta_t_tensor, mask_tensor, len(y_list)

# Ensure config values are set before use
def set_default_config_values():
    if config["max_features"] is None:
        config["max_features"] = 50  # Set default value if None
    if config["max_time_steps"] is None:
        config["max_time_steps"] = 200  # Set default value if None


# Function to extract y and t features from stay data
def extract_features(stay_id):
    stay_path = os.path.join(config["data_dir"], stay_id)
    features = sorted([f for f in os.listdir(stay_path) if 'ts' in f])
    y_list, t_list = [], []
    for feature_file in features:
        df = pd.read_csv(os.path.join(stay_path, feature_file))
        if df.dtypes[-1] == 'object':
            continue
        df.drop(df.columns[0], axis=1, inplace=True)
        if 'static' in feature_file:
            df.insert(0, 'charttime', 0)
        y_list.append(torch.tensor(df.iloc[:, 1].values).float().reshape(-1, 1))
        t_list.append(torch.tensor(df.iloc[:, 0].values).float().reshape(-1, 1))
    return y_list, t_list

def align_times(t_list):
    common_times = torch.unique(torch.cat(t_list))
    if config["max_time_steps"] is None:
        max_time_steps = min(config["time_steps_upper_limit"], len(common_times))
        config["max_time_steps"] = max_time_steps
    if len(common_times) > config["max_time_steps"]:
        common_times = common_times[:config["max_time_steps"]]
    elif len(common_times) < config["max_time_steps"]:
        pad_size = config["max_time_steps"] - len(common_times)
        common_times = torch.cat([common_times, common_times[-1:].expand(pad_size)])
    t_tensor = common_times.unsqueeze(0).expand(len(t_list), -1)
    return common_times, t_tensor

def align_y_and_masks(y_list, t_tensor, common_times):
    aligned_y, aligned_mask = [], []
    for y_array, t_array in zip(y_list, t_tensor):
        y_aligned = torch.zeros((config["max_time_steps"], 1))
        mask = torch.zeros_like(y_aligned)
        for i, t in enumerate(common_times):
            match_idx = (t_array == t).nonzero(as_tuple=True)[0]
            if match_idx.numel() > 0 and match_idx.item() < y_array.size(0):
                y_aligned[i] = y_array[match_idx]
                mask[i] = 1
        aligned_y.append(y_aligned)
        aligned_mask.append(mask)
    y_tensor = torch.stack(aligned_y, dim=2)
    mask_tensor = torch.stack(aligned_mask, dim=2)
    return pad_y_and_mask_tensors(y_tensor, mask_tensor, len(y_list))

def pad_y_and_mask_tensors(y_tensor, mask_tensor, num_nodes):
    if num_nodes < config["max_features"]:
        padding_size = config["max_features"] - num_nodes
        y_tensor = torch.nn.functional.pad(y_tensor, (0, padding_size))
        mask_tensor = torch.nn.functional.pad(mask_tensor, (0, padding_size))
    return y_tensor, mask_tensor

def calculate_delta_t(t_tensor, y_tensor):
    return utils.t_to_delta_t(t_tensor).unsqueeze(0).repeat(1, y_tensor.shape[0], y_tensor.shape[2])

# Main function to run the script
def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Generate dataset')
    args = parser.parse_args()
    config.update(vars(args))

    torch.random.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    stay_ids = sorted(os.listdir(config["data_dir"]))[:10]
    split_data = initialize_split_data()

    for stay_id in tqdm(stay_ids):
        edge_index, y_tensor, t_tensor, delta_t_tensor, mask_tensor, num_nodes = process_stay(stay_id)
        append_split_data(split_data, 'train', y_tensor, t_tensor, delta_t_tensor, mask_tensor)

    for split in ['train', 'val', 'test']:
        process_split_data(split_data, split)

    utils.save_data("g3_mimic_demo", config, split_data)
    print("Data saved successfully.")



if __name__ == "__main__":

    # Continue with the main function
    main()
