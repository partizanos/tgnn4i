import torch
import torch_geometric as ptg
import numpy as np
import argparse
import os
import pandas as pd
import utils
from tqdm import tqdm
import logging
import pdb  # Import the Python debugger

config = {
    "data_dir": "step_2_mimic_demo/",
    "output_dir": "dataset/g3_mimic_demo/",
    "max_time_steps": None,
    "max_features": None,
    "seed": 42,
    "time_steps_upper_limit": 400,  # Set an upper limit for max_time_steps
    "batch_size": 1  # Set an upper limit for max_time_steps
}

def create_fully_connected_graph(num_nodes):
    return torch.randint(0, num_nodes, (2, 50))

def pad_tensor(tensor, max_shape):
    print(f"Padding tensor of shape {tensor.shape} to max shape {max_shape}")
    
    # Adjust tensor shape to match max_shape
    tensor_shape = list(tensor.shape)
    while len(tensor_shape) < len(max_shape):
        tensor_shape.append(1)  # Add singleton dimensions to match max_shape

    print(f"Adjusted tensor shape: {tensor_shape}")
    
    # Initialize the padded tensor with zeros of the target max_shape
    padded_tensor = torch.zeros(max_shape)
    
    # Check if tensor exceeds any of the max dimensions
    exceeds_dims = [tensor_shape[i] > max_shape[i] for i in range(len(max_shape))]
    print(f"Exceeds max dimensions: {exceeds_dims}")
    
    # Compute slices for each dimension, ensure tensor's slice matches the tensor's dimension
    slices = tuple(slice(0, min(dim, max_dim)) for dim, max_dim in zip(tensor_shape, max_shape))
    
    print(f"Slices: {slices}, Padded tensor shape: {padded_tensor.shape}")
    
    try:
        # Assign the values from the original tensor to the corresponding slice in the padded tensor
        padded_tensor[slices] = tensor[slices[:tensor.ndimension()]]
    except Exception as e:
        print(f"Error during tensor padding: {e}")
        pdb.set_trace()  # Debugger breakpoint

    return padded_tensor

def process_stay(stay_id):
    stay_path = os.path.join(config["data_dir"], stay_id)
    features = sorted([f for f in os.listdir(stay_path) if 'ts' in f])
    
    y_list, t_list = [], []
    
    for feature_file in features:
        df = pd.read_csv(os.path.join(stay_path, feature_file))
        type_of_last_col = df.dtypes[-1]
        if type_of_last_col == 'object':
            continue
        
        df.drop(df.columns[0], axis=1, inplace=True)
        if 'static' in feature_file:
            df.insert(0, 'charttime', 0)
        
        feature_name = df.columns[1]
        y_array = torch.tensor(df[feature_name].values).float().reshape(-1, 1)
        time_col = df.columns[0]
        t_array = torch.tensor(df[time_col].values).float().reshape(-1, 1)
        
        y_list.append(y_array)
        t_list.append(t_array)

    common_times = torch.unique(torch.cat(t_list))
    
    if config["max_time_steps"] is None or len(common_times) < config["max_time_steps"]:
        pad_size = (config["max_time_steps"] or len(common_times)) - len(common_times)
        padding = common_times[-1:].expand(pad_size)
        common_times = torch.cat([common_times, padding])
    elif len(common_times) > config["max_time_steps"]:
        common_times = common_times[:config["max_time_steps"]]
    
    num_nodes = len(y_list)
    t_tensor = common_times.unsqueeze(0).expand(num_nodes, config["max_time_steps"] or len(common_times))
    
    aligned_y, aligned_mask = [], []

    for y_array, t_array in zip(y_list, t_list):
        y_aligned = torch.zeros((config["max_time_steps"] or len(common_times), 1))
        mask = torch.zeros_like(y_aligned)
        
        for i, t in enumerate(common_times):
            match_idx = (t_array == t).nonzero(as_tuple=True)[0]
            if match_idx.size(0) > 0:
                y_aligned[i] = y_array[match_idx]
                mask[i] = 1
        
        aligned_y.append(y_aligned)
        aligned_mask.append(mask)

    y_tensor = torch.stack(aligned_y, dim=2)
    mask_tensor = torch.stack(aligned_mask, dim=2)

    if config["max_features"] is not None and num_nodes < config["max_features"]:
        padding_size = config["max_features"] - num_nodes
        y_tensor = torch.nn.functional.pad(y_tensor, (0, padding_size))
        mask_tensor = torch.nn.functional.pad(mask_tensor, (0, padding_size))

    delta_t_tensor = utils.t_to_delta_t(t_tensor).unsqueeze(0).repeat(1, y_tensor.shape[0], y_tensor.shape[2])
    
    return create_fully_connected_graph(num_nodes), y_tensor, t_tensor, delta_t_tensor, mask_tensor, num_nodes

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Generate dataset')
    # Arguments setup
    args = parser.parse_args()
    config.update(vars(args))

    # Seed and setup
    _ = torch.random.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Data loading and processing
    stay_ids = sorted(os.listdir(config["data_dir"]))[:10]
    n_stays = len(stay_ids)
    n_val = n_test = int(0.2 * n_stays)
    n_train = n_stays - n_val - n_test

    train_stays = stay_ids[:n_train]
    val_stays = stay_ids[n_train:n_train + n_val]
    test_stays = stay_ids[n_train + n_val:]

    split_data = {'train': {'y': [], 't': [], 'delta_t': [], 'mask': []},
                  'val': {'y': [], 't': [], 'delta_t': [], 'mask': []},
                  'test': {'y': [], 't': [], 'delta_t': [], 'mask': []}}
    
    node_offset = 0

    if config["max_time_steps"] is None or config["max_features"] is None:
        logging.info("Calculating max_time_steps and max_features dynamically.")
        for stay_id in tqdm(stay_ids):
            features = sorted([f for f in os.listdir(os.path.join(config["data_dir"], stay_id)) if 'ts' in f])
            t_list = []
            
            for feature_file in features:
                df = pd.read_csv(os.path.join(config["data_dir"], stay_id, feature_file))
                df.drop(df.columns[0], axis=1, inplace=True)
                time_col = df.columns[0]
                t_array = torch.tensor(df[time_col].values).reshape(-1, 1)
                t_list.append(t_array)
            
            common_times = torch.unique(torch.cat(t_list))
            config["max_time_steps"] = min(max(config["max_time_steps"] or 0, common_times.size(0)), config["time_steps_upper_limit"])
            config["max_features"] = max(config["max_features"] or 0, len(features))

    all_edge = []
    for stay_id in tqdm(stay_ids):
        edge_index, y_tensor, t_tensor, delta_t_tensor, mask_tensor, num_nodes = process_stay(stay_id)
        edge_index = edge_index + node_offset
        node_offset += num_nodes
        all_edge.append(edge_index)
        
        if stay_id in train_stays:
            split_name = 'train'
        elif stay_id in val_stays:
            split_name = 'val'
        else:
            split_name = 'test'
        
        split_data[split_name]['y'].append(y_tensor)
        split_data[split_name]['t'].append(t_tensor)
        split_data[split_name]['delta_t'].append(delta_t_tensor)
        split_data[split_name]['mask'].append(mask_tensor)

    for split in ['train', 'val', 'test']:
        max_shape = (config["max_time_steps"], 1, config["max_features"], 1)
        
        # Padding y tensor
        split_data[split]['y'] = [pad_tensor(y.unsqueeze(-1), max_shape) for y in split_data[split]['y']]
        assert all(y.ndim == 4 for y in split_data[split]['y']), "y tensor must have 4 dimensions [time_steps, 1, features, 1]"
        
        # Reshape and pad t tensor
        split_data[split]['t'] = [pad_tensor(t.reshape(-1, 1), (t.shape[0], 1, 1)) for t in split_data[split]['t']]
        assert all(t.ndim == 3 for t in split_data[split]['t']), "t tensor must have 3 dimensions [time_steps, 1, 1]"
        
        # Padding delta_t tensor
        split_data[split]['delta_t'] = [pad_tensor(delta_t.unsqueeze(-1), max_shape) for delta_t in split_data[split]['delta_t']]
        assert all(delta_t.ndim == 4 for delta_t in split_data[split]['delta_t']), "delta_t tensor must have 4 dimensions [time_steps, 1, features, 1]"
        
        # Padding mask tensor
        split_data[split]['mask'] = [pad_tensor(mask.unsqueeze(-1), max_shape) for mask in split_data[split]['mask']]
        assert all(mask.ndim == 4 for mask in split_data[split]['mask']), "mask tensor must have 4 dimensions [time_steps, 1, features, 1]"

        split_data[split]['y'] = np.array(split_data[split]['y'])
        split_data[split]['t'] = np.array(split_data[split]['t'])
        split_data[split]['delta_t'] = np.array(split_data[split]['delta_t'])
        split_data[split]['mask'] = np.array(split_data[split]['mask'])


    save_dict = {}
    for set_name in ["train", "val", "test"]:
        save_dict[set_name] = {
            "y": np.array(split_data[set_name]['y']),
            "t": np.array(split_data[set_name]['t']),
            "delta_t": np.array(split_data[set_name]['delta_t']),
            "mask": np.array(split_data[set_name]['mask']),
        }
    save_dict["edge_index"] = torch.cat(all_edge, dim=1).numpy()

    ds_name = f"g3_mimic_demo"
    utils.save_data(ds_name, config, save_dict)

    print("Data saved successfully.")

    # Post-save validation
    print("\n[Post-save validation]: Testing data loading and processing")
    data_dict = utils.load_temporal_graph_data(ds_name, config["batch_size"])

    # Checking the data structure
    for subset in ['train', 'val', 'test']:
        print(f"\n--- Checking {subset} data ---")
        y = data_dict[subset]['y']
        t = data_dict[subset]['t']
        delta_t = data_dict[subset]['delta_t']
        mask = data_dict[subset]['mask']
        edge_index = data_dict['edge_index']

        print(f"y shape: {y.shape}, Expected: [batch_size, time_steps, features, 1]")
        print(f"t shape: {t.shape}, Expected: [batch_size, time_steps]")
        print(f"delta_t shape: {delta_t.shape}, Expected: [batch_size, features, time_steps]")
        print(f"mask shape: {mask.shape}, Expected: [batch_size, features, time_steps]")
        print(f"edge_index shape: {edge_index.shape}, Expected: [2, num_edges]")

    print("Data validation completed.")

if __name__ == "__main__":
    main()
