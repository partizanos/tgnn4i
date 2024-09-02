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
    "max_features": 17,
    "seed": 42,
    "time_steps_upper_limit": 400,
    "batch_size": 1,
    "graph_edges": 50
}

# Function to initialize split data dictionary
def initialize_split_data():
    return {'train': {'y': [], 't': [], 'delta_t': [], 'mask': []},
            'val': {'y': [], 't': [], 'delta_t': [], 'mask': []},
            'test': {'y': [], 't': [], 'delta_t': [], 'mask': []}}

# Function to append data to split data dictionary
def append_split_data(split_data, split, y_tensor, t_tensor, delta_t_tensor, mask_tensor):
    split_data[split]['y'].append(y_tensor)
    # import pdb; pdb.set_trace()
    # if t_tensor.ndim == 3:
    #     t_tensor= t_tensor.squeeze(-2).squeeze(-1)
    # if t_tensor.ndim == 2:
    #     t_tensor= t_tensor.squeeze(-1)
    # t_tensor.squeeze(2)
    # t_tensor=t_tensor.squeeze(1)
    split_data[split]['t'].append(t_tensor)
    
    split_data[split]['delta_t'].append(delta_t_tensor)
    split_data[split]['mask'].append(mask_tensor)

   

# Function to create a fully connected graph
def create_fully_connected_graph(num_nodes):
    return torch.randint(0, num_nodes, (2, config["graph_edges"]))

def pad_tensor(tensor, max_shape):
    tensor_shape = list(tensor.shape)
    
    # Ensure all elements in max_shape are integers and defined
    max_shape = tuple(int(dim) if dim is not None else tensor_shape[i] for i, dim in enumerate(max_shape))
    
    # Check if tensor shape is already matching max_shape
    if tensor_shape == list(max_shape):
        return tensor

    # # Check for incompatible dimensions and handle gracefully
    # if any(ts > ms for ts, ms in zip(tensor_shape, max_shape)):
    #     raise ValueError(f"Cannot pad tensor of shape {tensor_shape} to max_shape {max_shape}. Tensor dimensions must not exceed the max_shape dimensions.")

    # Adjust dimensions if necessary by unsqueezing or squeezing
    count_missing_dims = abs(len(max_shape) - tensor.ndim)
    if len(max_shape) > tensor.ndim:
        for _ in range(count_missing_dims):
            tensor = tensor.unsqueeze(-1)
    elif tensor.ndim > len(max_shape):
        for _ in range(count_missing_dims):
            tensor = tensor.squeeze(-1)

    # Create the padded tensor and apply the padding only if necessary
    padded_tensor = torch.zeros(max_shape)
    
    # Adjust the slices to match the tensor's dimensions
    slices = tuple(slice(0, min(dim, max_dim)) for dim, max_dim in zip(tensor.shape, max_shape))
    
    try:   
        padded_tensor[slices] = tensor[slices]
    except Exception as e:
        logging.error(f"Error in padding tensor: {e}")
        logging.error(f"Slices: {slices}")
        logging.error(f"Tensor shape: {tensor.shape}, padded tensor shape: {padded_tensor.shape}")
        raise

    return padded_tensor

# Function to process a single stay and extract features
def process_stay(stay_id):
    y_list, t_list = extract_features(stay_id)
    common_times, t_tensor = align_times(t_list)
    # import pdb; pdb.set_trace()
    y_tensor, mask_tensor = align_y_and_masks(y_list, t_tensor, common_times)
    # import pdb; pdb.set_trace()
    delta_t_tensor = calculate_delta_t(t_tensor, y_tensor)
    # import pdb; pdb.set_trace()
    assert t_tensor.ndim == 2, "t_tensor must have 2 dimensions"
    torch.Size([87, 200])
    return create_fully_connected_graph(len(y_list)), y_tensor, t_tensor, delta_t_tensor, mask_tensor, len(y_list)

# Function to extract y and t features from stay data
def extract_features(stay_id):
    stay_path = os.path.join(config["data_dir"], stay_id)
    features = sorted([f for f in os.listdir(stay_path) 
                       if ('ts' in f) or ('static' in f)
                    ])
    y_list, t_list = [], []
    for feature_file in features:
        df = pd.read_csv(os.path.join(stay_path, feature_file))
        df.drop(df.columns[0], axis=1, inplace=True)
        if 'static' in feature_file:
            df.insert(0, 'charttime', 0)
        if df.dtypes[-1] == 'object':
            continue
        try:
            y_list.append(torch.tensor(df.iloc[:, 1].values).float().reshape(-1, 1))
            t_list.append(torch.tensor(df.iloc[:, 0].values).float().reshape(-1, 1))
        except Exception as e:
            import pdb; pdb.set_trace()
    return y_list, t_list

def align_times(t_list):
    common_times = torch.unique(torch.cat(t_list))
    # if config["max_time_steps"] is None: TODO automatic find max_time_steps
    #     max_time_steps = min(config["time_steps_upper_limit"], len(common_times))
    #     config["max_time_steps"] = max_time_steps
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
            if match_idx.numel() > 0:
                first_match_idx = match_idx[0]  # Handle the first match
                if first_match_idx.item() < y_array.size(0):
                    y_aligned[i] = y_array[first_match_idx]
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
    # Calculate the time differences between consecutive time points along the time dimension
    # Assumption: t_tensor is shaped (batch_size, num_time_steps, 1)
    # and y_tensor is shaped (num_time_steps, num_nodes, feature_dim)
    
    # Compute time differences (delta_t) along the time dimension (axis 1)
    delta_t = t_tensor[:, 1:] - t_tensor[:, :-1]

    # Append a zero at the beginning to maintain the same shape as t_tensor
    delta_t = torch.cat([torch.zeros_like(delta_t[:, :1]), delta_t], dim=1)

    # Adjust the shape to match y_tensor's time and node dimensions
    delta_t = delta_t.unsqueeze(2).repeat(1, 1, y_tensor.shape[2])
    
    return delta_t
def convert_to_tensor(stayid_2_stayDict, stay_id):
    y = torch.stack(stayid_2_stayDict[stay_id]['y_tensor'] )
    t = torch.stack(stayid_2_stayDict[stay_id]['t_tensor'] )
    delta_t = torch.stack(stayid_2_stayDict[stay_id]['delta_t_tensor'] )
    mask = torch.stack(stayid_2_stayDict[stay_id]['mask_tensor'] )
    return y, t, delta_t, mask
# Main function to run the script

def append_stays(stayid_2_stayDict, split_idx):
    all_data=[]
    all_y_list=[]
    all_t_list=[]
    all_delta_t_list=[]
    all_mask_list=[]
    all_edge_index_list=[]
    for stay_id in split_idx:
        all_data.append(stayid_2_stayDict[stay_id])
        all_y_list.append(stayid_2_stayDict[stay_id]['y_tensor'])
        all_t_list.append(stayid_2_stayDict[stay_id]['t_tensor'])
        all_delta_t_list.append(stayid_2_stayDict[stay_id]['delta_t_tensor'])
        all_mask_list.append(stayid_2_stayDict[stay_id]['mask_tensor'])
        all_edge_index_list.append(stayid_2_stayDict[stay_id]['edge_index'])


    def pad_or_trim_tensor(tensor, max_time_steps):
        current_time_steps = tensor.shape[0]
        
        if current_time_steps > max_time_steps:
            # Trim to the first 100 time steps
            tensor = tensor[:max_time_steps]
        elif current_time_steps < max_time_steps:
            # Pad with zeros to reach 100 time steps
            padding_size = max_time_steps - current_time_steps
            padding_shape = (padding_size,) + tensor.shape[1:]
            padding = torch.zeros(padding_shape)
            tensor = torch.cat([tensor, padding], dim=0)
        
        return tensor

    max_time_steps = 20
    # Apply to all tensors in the lists
    all_y_list = [pad_or_trim_tensor(y, max_time_steps) for y in all_y_list]
    all_t_list = [pad_or_trim_tensor(t, max_time_steps) for t in all_t_list]
    all_delta_t_list = [pad_or_trim_tensor(delta_t, max_time_steps) for delta_t in all_delta_t_list]
    all_mask_list = [pad_or_trim_tensor(mask, max_time_steps) for mask in all_mask_list]

    # trim also number of featues 3ed (counting from 9) dimension
    # import pdb; pdb.set_trace()
    all_y_list = [y[:,:,:config["max_features"]] for y in all_y_list]
    # all_t_list = [t[:,:,:config["max_features"]] for t in all_t_list]
    all_delta_t_list = [delta_t[:,:,:config["max_features"]] for delta_t in all_delta_t_list]
    all_mask_list = [mask[:,:,:config["max_features"]] for mask in all_mask_list]
    # all_edge_index_list = [edge_index for edge_index in all_edge_index_list]



    y_tensor = torch.stack(all_y_list)
    t_tensor = torch.stack(all_t_list)
    delta_t_tensor = torch.stack(all_delta_t_list)
    mask_tensor = torch.stack(all_mask_list)
    edge_index = torch.stack(all_edge_index_list)
    return y_tensor, t_tensor, delta_t_tensor, mask_tensor, edge_index


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Generate dataset')
    args = parser.parse_args()
    config.update(vars(args))

    torch.random.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    stay_ids = sorted(os.listdir(config["data_dir"]))[:10]
    split_data = initialize_split_data()
    stayid_2_stayDict={}
    all_edge_index_list = []
    for stay_id in tqdm(stay_ids):
        edge_index, y_tensor, t_tensor, delta_t_tensor, mask_tensor, num_nodes = process_stay(stay_id)
        stayid_2_stayDict[stay_id] = {'edge_index': edge_index, 'y_tensor': y_tensor, 't_tensor': t_tensor, 'delta_t_tensor': delta_t_tensor, 'mask_tensor': mask_tensor}
        all_edge_index_list.append(edge_index)
    # random split 
    n = len(stay_ids)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx = idx[:int(0.8*n)]
    val_idx = idx[int(0.8*n):int(0.9*n)]
    test_idx = idx[int(0.9*n):]
    for split_key, idxes in zip(['train', 'val', 'test'],[train_idx, val_idx, test_idx]):
        # combine stays with idx of same set train, val, test int split variable 
        split_idx = [ stay_ids[i] for i in idxes] 
        # store split in split_data according to idx-es
        subset_stayid_2_stayDict = {k: stayid_2_stayDict[k] for k in split_idx}
        y_tensor, t_tensor, delta_t_tensor, mask_tensor, edge_index = append_stays(subset_stayid_2_stayDict, split_idx)
        append_split_data(split_data, 'train', y_tensor, t_tensor, delta_t_tensor, mask_tensor)

    split_data['edge_index'] = torch.stack(all_edge_index_list) # TESTI TODO that edge index is on top key
    utils.save_data("g3_mimic_demo", config, split_data)
    print("Data saved successfully.")

    ##### 
    from main import get_config
    config_main = get_config()

    # Set all random seeds
    np.random.seed(config_main["seed"])
    torch.manual_seed(config_main["seed"])

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # For reproducability on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    # Load data
    # torch.multiprocessing.set_sharing_strategy('file_system') # Fix for num_workers > 0 # TODO undo in HPC, laptop env crashes python in small RAM context
    train_loader, val_loader, test_loader = utils.load_temporal_graph_data(
            config_main["dataset"], config_main["batch_size"],
            compute_hop_mask=(config_main["state_updates"] == "hop"), L_hop=config_main["gru_gnn"])
    # test iteration one line
    [ batch for batch in train_loader]
if __name__ == "__main__":
    main()
