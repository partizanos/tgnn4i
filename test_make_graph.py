import unittest
import torch
from make_graph_3 import (
    align_times,
    pad_tensor,
    process_split_data,
    config,
    create_fully_connected_graph,
    extract_features,
    process_stay,
    initialize_split_data,
    append_split_data
)

class TestMakeGraph(unittest.TestCase):

    def test_align_times_small(self):
        t_list_small = [torch.tensor([1, 2, 3]).float().reshape(-1, 1),
                        torch.tensor([1, 2, 4]).float().reshape(-1, 1)]
        
        common_times_small, t_tensor_small = align_times(t_list_small)
        self.assertEqual(t_tensor_small.shape, (2, config["max_time_steps"]), "t_tensor_small shape mismatch")
        self.assertEqual(len(common_times_small), config["max_time_steps"], "common_times_small length mismatch")
        print("Test with small t_list passed.")
    
    def test_align_times_large(self):
        t_list_large = [torch.arange(1, 250).float().reshape(-1, 1),
                        torch.arange(1, 250).float().reshape(-1, 1)]
        
        common_times_large, t_tensor_large = align_times(t_list_large)
        self.assertEqual(t_tensor_large.shape, (2, config["max_time_steps"]), "t_tensor_large shape mismatch")
        self.assertEqual(len(common_times_large), config["max_time_steps"], "common_times_large length mismatch")
        print("Test with large t_list passed.")

    def test_pad_tensor(self):
        # Test case 1: Tensor with smaller dimensions than max_shape
        tensor = torch.randn(50, 1)
        max_shape = (100, 1, 1)
        padded_tensor = pad_tensor(tensor, max_shape)
        self.assertEqual(padded_tensor.shape, torch.Size(max_shape), f"Expected {max_shape}, got {padded_tensor.shape}")
        
        # Test case 2: Tensor with dimensions matching max_shape
        tensor = torch.randn(100, 1, 1)
        max_shape = (100, 1, 1)
        padded_tensor = pad_tensor(tensor, max_shape)
        self.assertEqual(padded_tensor.shape, torch.Size(max_shape), f"Expected {max_shape}, got {padded_tensor.shape}")
        
        # Test case 3: Tensor with more dimensions than max_shape
        tensor = torch.randn(100, 2, 3)
        max_shape = (100, 1, 1)
        with self.assertRaises(RuntimeError):
            pad_tensor(tensor, max_shape)

        print("All tests for pad_tensor passed.")
    
    def test_process_split_data(self):
        # Create mock split_data dictionary
        split_data = {
            'train': {
                'y': [torch.randn(50, 1) for _ in range(5)],
                't': [torch.randn(50) for _ in range(5)],
                'delta_t': [torch.randn(50, 1) for _ in range(5)],
                'mask': [torch.randn(50, 1) for _ in range(5)]
            }
        }
        
        # Run process_split_data
        process_split_data(split_data, 'train')
        
        for y_tensor in split_data['train']['y']:
            self.assertEqual(y_tensor.shape, torch.Size((config["max_time_steps"], 1, config["max_features"], 1)),
                             f"Expected y_tensor shape to be {(config['max_time_steps'], 1, config['max_features'], 1)}, got {y_tensor.shape}")
        
        for t_tensor in split_data['train']['t']:
            self.assertEqual(t_tensor.shape, torch.Size((config["max_time_steps"], 1, 1)),
                             f"Expected t_tensor shape to be {(config['max_time_steps'], 1, 1)}, got {t_tensor.shape}")
        
        for delta_t_tensor in split_data['train']['delta_t']:
            self.assertEqual(delta_t_tensor.shape, torch.Size((config["max_time_steps"], 1, config["max_features"], 1)),
                             f"Expected delta_t_tensor shape to be {(config['max_time_steps'], 1, config['max_features'], 1)}, got {delta_t_tensor.shape}")
        
        for mask_tensor in split_data['train']['mask']:
            self.assertEqual(mask_tensor.shape, torch.Size((config["max_time_steps"], 1, config["max_features"], 1)),
                             f"Expected mask_tensor shape to be {(config['max_time_steps'], 1, config['max_features'], 1)}, got {mask_tensor.shape}")

        # Assuming edge_index and edge_weight are part of split_data or generated
        # Add tests here if applicable

        print("All tests for process_split_data passed.")

    def test_create_fully_connected_graph(self):
        num_nodes = 5
        edge_index = create_fully_connected_graph(num_nodes)
        self.assertEqual(edge_index.shape, (2, config["graph_edges"]),
                         f"Expected edge_index shape to be (2, {config['graph_edges']}), got {edge_index.shape}")
        self.assertTrue((edge_index < num_nodes).all(),
                        "All edge indices should be less than num_nodes")
        print("Test for create_fully_connected_graph passed.")

    def test_extract_features(self):
        # Assume we have some mock data in the directory `step_2_mimic_demo/201006/`
        config["data_dir"] = "step_2_mimic_demo/"
        stay_id = "201006"
        y_list, t_list = extract_features(stay_id)
        self.assertIsInstance(y_list, list, "y_list should be a list.")
        self.assertIsInstance(t_list, list, "t_list should be a list.")
        self.assertGreater(len(y_list), 0, "y_list should not be empty.")
        self.assertGreater(len(t_list), 0, "t_list should not be empty.")
        self.assertTrue(all(isinstance(y, torch.Tensor) for y in y_list),
                        "All elements in y_list should be torch.Tensors.")
        self.assertTrue(all(isinstance(t, torch.Tensor) for t in t_list),
                        "All elements in t_list should be torch.Tensors.")
        print("Test for extract_features passed.")

    def test_process_stay(self):
        config["data_dir"] = "step_2_mimic_demo/"
        stay_id = "201006"
        edge_index, y_tensor, t_tensor, delta_t_tensor, mask_tensor, num_nodes = process_stay(stay_id)
        self.assertIsInstance(edge_index, torch.Tensor, "edge_index should be a torch.Tensor.")
        self.assertIsInstance(y_tensor, torch.Tensor, "y_tensor should be a torch.Tensor.")
        self.assertIsInstance(t_tensor, torch.Tensor, "t_tensor should be a torch.Tensor.")
        self.assertIsInstance(delta_t_tensor, torch.Tensor, "delta_t_tensor should be a torch.Tensor.")
        self.assertIsInstance(mask_tensor, torch.Tensor, "mask_tensor should be a torch.Tensor.")
        self.assertEqual(num_nodes, len(y_tensor[0, 0, :]), "num_nodes should match the number of features.")
        print("Test for process_stay passed.")

    def test_initialize_split_data(self):
        split_data = initialize_split_data()
        self.assertIn('train', split_data, "split_data should contain 'train' key.")
        self.assertIn('val', split_data, "split_data should contain 'val' key.")
        self.assertIn('test', split_data, "split_data should contain 'test' key.")
        self.assertIsInstance(split_data['train'], dict, "'train' should be a dictionary.")
        self.assertEqual(split_data['train']['y'], [], "'train' 'y' key should start as an empty list.")
        self.assertEqual(split_data['train']['t'], [], "'train' 't' key should start as an empty list.")
        self.assertEqual(split_data['train']['delta_t'], [], "'train' 'delta_t' key should start as an empty list.")
        self.assertEqual(split_data['train']['mask'], [], "'train' 'mask' key should start as an empty list.")
        print("Test for initialize_split_data passed.")

    def test_append_split_data(self):
        split_data = initialize_split_data()
        y_tensor = torch.randn(config["max_time_steps"], 1, config["max_features"], 1)
        t_tensor = torch.randn(config["max_time_steps"], 1, 1)
        delta_t_tensor = torch.randn(config["max_time_steps"], 1, config["max_features"], 1)
        mask_tensor = torch.randn(config["max_time_steps"], 1, config["max_features"], 1)

        append_split_data(split_data, 'train', y_tensor, t_tensor, delta_t_tensor, mask_tensor)

        self.assertEqual(len(split_data['train']['y']), 1, "'train' 'y' key should have 1 entry.")
        self.assertEqual(split_data['train']['y'][0].shape, y_tensor.shape, "'train' 'y' tensor shape mismatch.")
        self.assertEqual(split_data['train']['t'][0].shape, t_tensor.shape, "'train' 't' tensor shape mismatch.")
        self.assertEqual(split_data['train']['delta_t'][0].shape, delta_t_tensor.shape, "'train' 'delta_t' tensor shape mismatch.")
        self.assertEqual(split_data['train']['mask'][0].shape, mask_tensor.shape, "'train' 'mask' tensor shape mismatch.")
        print("Test for append_split_data passed.")
if __name__ == "__main__":
    unittest.main()
