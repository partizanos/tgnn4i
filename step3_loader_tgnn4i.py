from tqdm import tqdm 
import pandas as pd
import torch
import torch_geometric as ptg
import numpy as np
import argparse
import os
import pdb  # Import Python debugger

import utils  # Make sure utils.py is in the same directory or accessible path
import visualization as vis  # Assuming this is another helper script
from models.gru_graph_model import GRUGraphModel
from tueplots import axes, bundles

from config import TIME_VARS_MAP_DATASET_LIST_KEYS,PRIMARY_KEY_MAP


ordinal_encoding= {}
# Argument parser
parser = argparse.ArgumentParser(description='Generate dataset')

parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--plot", type=int, default=0,
                    help="If plots should be made during generation (and number of plots)")
parser.add_argument("--max_nodes_plot", type=int, default=5,
                    help="Maximum number of nodes to plot predictions for")

# Graph construction
parser.add_argument("--n_nodes", type=int, default=10,
                    help="Number of nodes in graph")
parser.add_argument("--graph_alg", type=str, default="delaunay",
                    help="Algorithm to use for constructing graph")
parser.add_argument("--n_neighbors", type=int, default=5,
                    help="Amount of neighbors to include in k-nn graph generation")
parser.add_argument("--data_dir", type=str, default="path/to/data_directory",
                    help="Directory containing the data files")

# Signal construction
parser.add_argument("--T", type=float, default=2.0,
                    help="Time horizon")
parser.add_argument("--neighbor_weight", type=float, default=1.0,
                    help="Weighting on the influence of neighbor signals")
parser.add_argument("--lag", type=float, default=0.1,
                    help="Lag of neighbor influence")
parser.add_argument("--noise_std", type=float, default=0.05,
                    help="Std.-dev. of added noise")
parser.add_argument("--batch_size", type=int, default=256,
                    help="Batch size used for batched computations")

# Dataset stats
parser.add_argument("--n_t", type=int, default=20,
                    help="Number of time points to evaluate at")
parser.add_argument("--obs_nodes", type=str, default="0.25",
                    help="Percentage of nodes observed at each timestep (in [0,1] or 'single')")
parser.add_argument("--n_train", type=int, default=100,
                    help="Number of data points (time series) in training set")
parser.add_argument("--n_val", type=int, default=50,
                    help="Number of data points (time series) in validation set")
parser.add_argument("--n_test", type=int, default=50,
                    help="Number of data points (time series) in test set")

config = vars(parser.parse_args())

# Set random seed
torch.random.manual_seed(config["seed"])
np.random.seed(config["seed"])
dataset = 'mimic_demo' # TODO parse from argsparse
# Load data from CSV files
def load_data_from_csv(directory):
    """
    Load data from CSV files in the specified directory and return a dictionary.
    Each key corresponds to a file and each value is a DataFrame with time steps as rows.
    """
    data = {}
    for stay_id in tqdm(os.listdir(directory)):
        stay_path = os.path.join(directory, stay_id)
        stay_list = os.listdir(stay_path)
        if len(stay_list) == 0:
            print(f"Warning: {stay_path} is empty and will be skipped.")
            continue
        for feature_name in stay_list:
            data[stay_id] = {}
            feature_path = os.path.join(stay_path, feature_name)
            # import pdb ; pdb.set_trace() #
            if feature_name.endswith('.csv'):
                # import pdb ; pdb.set_trace() 
                ordinal_encoding["_".join(feature_name.split("_")[2:])] = len(ordinal_encoding) # TODO,when mimic_demo removed replace with 2 -> 2 mimic_demo add one more '_' in the  the name
                if feature_path.endswith('.csv'):
                    try:
                        df = pd.read_csv(feature_path)
                        # Ensure DataFrame is not empty
                        if not df.empty:
                            data[stay_id][feature_name] = df
                        else:
                            print(f"Warning: {feature_path} is empty and will be skipped.")
                    except Exception as e:
                        import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
    return data



def common_step(df):
    # drop column that contains primary key 
    index_pkey= [i for i in range(len(df.columns)) if PRIMARY_KEY_MAP[dataset] in df.columns[i]][0]
    df.drop(df.columns[index_pkey], axis=1, inplace=True)

    return df 


def extract_from_static_data(df):
    df = common_step(df)
    return df


def extract_from_ts_data(df):
    df = common_step(df)
    # find time column among the dataset list of TIME_VARS_MAP_DATASET_LIST_KEYS[dataset]
    time_var_names = TIME_VARS_MAP_DATASET_LIST_KEYS[dataset]
    index_tcol = [i for i in range(len(df.columns)) if df.columns[i] in time_var_names]
    time_tensor = df[df.columns[index_tcol]]
    return df, time_tensor

def extract_from_ws_data(df):
    return extract_from_ts_data(df)

map_files_processing = {
    'static': extract_from_static_data,
    'ts': extract_from_ts_data,
    'ws': extract_from_ws_data
}

# Extract node features from data
def extract_node_features(data_dict):
    """
    Extract node features from the data dictionary.
    Each node corresponds to a time step with its features.
    """
    time_feature_list = []
    node_features_list = []
    for filename, icustay_id_name in tqdm(data_dict.items()):
        icustay_id_path = os.path.join(config["data_dir"], filename)
        
        for feature in os.listdir(icustay_id_path):

            feature_path = os.path.join(icustay_id_path, feature)
            df = pd.read_csv(feature_path)
            # is static in feature_path:
            if 'static' in feature_path:
                fn = map_files_processing['static'] 
            elif 'ts' in feature_path:
                fn = map_files_processing['ts']
            elif 'ws' in feature_path:
                fn = map_files_processing['ws']
                index_tcol= index_tcol[0]
            import pdb; pdb.set_trace()
            # new_node_feat, new_temp_feat = 
            fn(df)
            node_features_list.append(new_node_feat)
            
            # this time colum is used for the edges only 
            time_feature_list.append(new_temp_feat)

    if not node_features_list:
        pdb.set_trace()  # Enter debugger if the list is empty

    node_features = torch.cat(node_features_list, dim=0)  # Concatenate all features
    return node_features, time_feature_list

# Create graph for one sample
def create_graph(node_features):
    """
    Create a graph with node features and generate edges using a graph algorithm.
    """
    # Node positions (can be random or based on some other criteria)
    pos = torch.rand(node_features.size(0), 2)

    # Construct edges
    edge_index = None
    if config["graph_alg"] == "delaunay":
        edge_index = ptg.transforms.Delaunay()(ptg.data.Data(pos=pos)).edge_index
    elif config["graph_alg"] == "knn":
        edge_index = ptg.transforms.KNNGraph(k=config["n_neighbors"], force_undirected=True)(ptg.data.Data(pos=pos)).edge_index
    else:
        raise ValueError("Unknown graph algorithm")

    # Create PyTorch Geometric Data object
    graph_data = ptg.data.Data(x=node_features, pos=pos, edge_index=edge_index)
    return graph_data

# Load data
data_dict = load_data_from_csv(config["data_dir"])
assert len(data_dict.keys()) >0, f"No data loaded instead: {data_dict.keys()}"
node_features, time_features = extract_node_features(data_dict)

# Generate graph
print("Generating graph ...")
graph_data = create_graph(node_features)

if config["plot"]:
    vis.plot_graph(graph_data, show=True)  # Assuming vis is a visualization utility

# Prepare datasets
n_samples = config["n_train"] + config["n_val"] + config["n_test"]
print(f"Number of samples: {n_samples}")

# Function to split data into train, validation, and test sets
def split_data(graph_data, n_train, n_val, n_test):
    total_samples = n_train + n_val + n_test
    indices = np.arange(graph_data.x.size(0))  # Total number of nodes
    np.random.shuffle(indices)

    # Split indices
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Prepare PyTorch Geometric data objects for each split
    train_data = ptg.data.Data(x=graph_data.x[train_idx], edge_index=graph_data.edge_index, pos=graph_data.pos[train_idx])
    val_data = ptg.data.Data(x=graph_data.x[val_idx], edge_index=graph_data.edge_index, pos=graph_data.pos[val_idx])
    test_data = ptg.data.Data(x=graph_data.x[test_idx], edge_index=graph_data.edge_index, pos=graph_data.pos[test_idx])

    return train_data, val_data, test_data

# Split data
train_data, val_data, test_data = split_data(graph_data, config["n_train"], config["n_val"], config["n_test"])

# Save datasets
ds_name = f"periodic_{config['n_nodes']}_{config['obs_nodes']}_{config['seed']}"
save_dict = {
    "train": train_data,
    "val": val_data,
    "test": test_data,
    "edge_index": graph_data.edge_index.numpy()
}
utils.save_data(ds_name, config, save_dict)

print("Data saved")

# python step3_loader_tgnn4i.py --data_dir step_2_mimic_demo/