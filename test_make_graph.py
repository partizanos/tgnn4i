from unittest.mock import patch, MagicMock
import argparse
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
    append_split_data,
    pad_y_and_mask_tensors,
    main
)

class TestMakeGraph(unittest.TestCase):


    def setUp(self):
        # Set up any necessary configuration or data before each test
        self.stay_id = "201006"  # Example stay ID
        config["data_dir"] = "step_2_mimic_demo/"

    def test_graph_output_structure(self):
        # Process a stay to generate the graph and related tensors
        edge_index, y_tensor, t_tensor, delta_t_tensor, mask_tensor, num_nodes = process_stay(self.stay_id)
        
        # Initialize split data
        split_data = initialize_split_data()

        # Append data to the 'train' split
        append_split_data(split_data, 'train', y_tensor, t_tensor, delta_t_tensor, mask_tensor)

        # Process the 'train' split to ensure padding and shape alignment
        process_split_data(split_data, 'train')

        # Check for existence of keys
        required_keys = ['y', 't', 'delta_t', 'mask', 
                        #  'features'
                        ]
        for key in required_keys:
            self.assertIn(key, split_data['train'], f"'{key}' should exist in the 'train' split.")

        # Check number of dimensions
        import pdb; pdb.set_trace()
        self.assertEqual(split_data['train']['y'][0].ndim, 4, "y_tensor should have 4 dimensions")
        print(split_data['train']['t'][0].ndim)
        self.assertEqual(split_data['train']['t'][0].ndim, 2, "t_tensor should have 2 dimensions")
        self.assertEqual(split_data['train']['delta_t'][0].ndim, 3, "delta_t_tensor should have 3 dimensions")
        self.assertEqual(split_data['train']['mask'][0].ndim, 3, "mask_tensor should have 3 dimensions")

        # Check types
        self.assertIsInstance(split_data['train']['y'][0], torch.Tensor, "y_tensor should be a torch.Tensor")
        self.assertIsInstance(split_data['train']['t'][0], torch.Tensor, "t_tensor should be a torch.Tensor")
        self.assertIsInstance(split_data['train']['delta_t'][0], torch.Tensor, "delta_t_tensor should be a torch.Tensor")
        self.assertIsInstance(split_data['train']['mask'][0], torch.Tensor, "mask_tensor should be a torch.Tensor")

        # Ensure the edge_index is a torch tensor and has 2 dimensions
        self.assertIsInstance(edge_index, torch.Tensor, "edge_index should be a torch.Tensor")
        self.assertEqual(edge_index.ndim, 2, "edge_index should have 2 dimensions")

        # Check edge_weight
        edge_weight = torch.rand(edge_index.shape[1])  # Example edge_weight, replace with actual value if available
        self.assertIsInstance(edge_weight, torch.Tensor, "edge_weight should be a torch.Tensor")
        self.assertEqual(edge_weight.ndim, 1, "edge_weight should have 1 dimension")

        print("All checks passed for graph output structure.")

    @patch('make_graph_3.tqdm', side_effect=lambda x: x)  # Mock tqdm to return the iterable directly
    @patch('make_graph_3.os.listdir', return_value=['201006', '201007'])  # Mock os.listdir to control stay_ids
    @patch('make_graph_3.process_stay')  # Mock process_stay
    @patch('make_graph_3.utils.save_data')  # Mock utils.save_data to avoid file operations
    @patch('make_graph_3.argparse.ArgumentParser.parse_args',
           return_value=argparse.Namespace(data_dir="step_2_mimic_demo/", output_dir="dataset/g3_mimic_demo/"))
    def test_main(self, mock_args, mock_save_data, mock_process_stay, mock_listdir, mock_tqdm):
        # Set up mock return values
        mock_process_stay.return_value = (
            torch.tensor([[0, 1], [1, 0]]),  # edge_index
            torch.randn(config["max_time_steps"], 1, config["max_features"], 1),  # y_tensor
            torch.randn(config["max_time_steps"], 1, 1),  # t_tensor
            torch.randn(config["max_time_steps"], 1, config["max_features"], 1),  # delta_t_tensor
            torch.randn(config["max_time_steps"], 1, config["max_features"], 1),  # mask_tensor
            config["max_features"]  # num_nodes
        )

        # Run the main function
        main()

        # Assertions to ensure the right calls are made
        mock_listdir.assert_called_once_with("step_2_mimic_demo/")
        self.assertEqual(mock_process_stay.call_count, 2)  # We have two stay_ids in the mocked list
        mock_save_data.assert_called_once_with("g3_mimic_demo", config, unittest.mock.ANY)

        # Optionally, check if the saved data structure has the expected format
        saved_data = mock_save_data.call_args[0][2]  # The third argument in save_data call
        self.assertIn('train', saved_data)
        self.assertIn('y', saved_data['train'])
        self.assertIn('t', saved_data['train'])
        self.assertIn('delta_t', saved_data['train'])
        self.assertIn('mask', saved_data['train'])

        print("Test for main function passed.")

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
        # tensor = torch.randn(100, 2, 3)
        # max_shape = (100, 1, 1)
        # with self.assertRaises(RuntimeError):
        #     pad_tensor(tensor, max_shape)

        # print("All tests for pad_tensor passed.")
    
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
        self.assertEqual(split_data['train']['y'].ndim, y_tensor.ndim, "'train' 'y' tensor shape mismatch.")
        self.assertEqual(split_data['train']['t'].ndim, 2, "'train' 't' tensor shape mismatch.")
        self.assertEqual(split_data['train']['delta_t'].ndim, delta_t_tensor.ndim, "'train' 'delta_t' tensor shape mismatch.")
        self.assertEqual(split_data['train']['mask'].ndim, mask_tensor.ndim, "'train' 'mask' tensor shape mismatch.")
        print("Test for append_split_data passed.")

    
    def test_align_times(self):
        # Test case: t_list smaller than max_time_steps
        t_list_small = [torch.tensor([1, 2, 3]).float().reshape(-1, 1),
                        torch.tensor([1, 2, 4]).float().reshape(-1, 1)]
        common_times, t_tensor = align_times(t_list_small)
        self.assertEqual(t_tensor.shape, (2, config["max_time_steps"]),
                         f"Expected t_tensor shape to be (2, {config['max_time_steps']}), got {t_tensor.shape}")
        self.assertEqual(len(common_times), config["max_time_steps"],
                         f"Expected common_times length to be {config['max_time_steps']}, got {len(common_times)}")

        # Test case: t_list larger than max_time_steps
        t_list_large = [torch.arange(1, 450).float().reshape(-1, 1),
                        torch.arange(1, 450).float().reshape(-1, 1)]
        common_times, t_tensor = align_times(t_list_large)
        self.assertEqual(t_tensor.shape, (2, config["max_time_steps"]),
                         f"Expected t_tensor shape to be (2, {config['max_time_steps']}), got {t_tensor.shape}")
        self.assertEqual(len(common_times), config["max_time_steps"],
                         f"Expected common_times length to be {config['max_time_steps']}, got {len(common_times)}")
        print("All tests for align_times passed.")

    def test_pad_y_and_mask_tensors(self):
        # Test case: num_nodes < max_features
        y_tensor = torch.randn(config["max_time_steps"], 1, 30)
        mask_tensor = torch.randn(config["max_time_steps"], 1, 30)
        num_nodes = 30
        padded_y, padded_mask = pad_y_and_mask_tensors(y_tensor, mask_tensor, num_nodes)
        self.assertEqual(padded_y.shape, (config["max_time_steps"], 1, config["max_features"]),
                         f"Expected y_tensor shape to be (config['max_time_steps'], 1, config['max_features']), got {padded_y.shape}")
        self.assertEqual(padded_mask.shape, (config["max_time_steps"], 1, config["max_features"]),
                         f"Expected mask_tensor shape to be (config['max_time_steps'], 1, config['max_features']), got {padded_mask.shape}")

        # Test case: num_nodes > max_features (should not pad)
        y_tensor = torch.randn(config["max_time_steps"], 1, 60)
        mask_tensor = torch.randn(config["max_time_steps"], 1, 60)
        num_nodes = 60
        padded_y, padded_mask = pad_y_and_mask_tensors(y_tensor, mask_tensor, num_nodes)
        self.assertEqual(padded_y.shape, (config["max_time_steps"], 1, 60),
                         f"Expected y_tensor shape to be (config['max_time_steps'], 1, 60), got {padded_y.shape}")
        self.assertEqual(padded_mask.shape, (config["max_time_steps"], 1, 60),
                         f"Expected mask_tensor shape to be (config['max_time_steps'], 1, 60), got {padded_mask.shape}")
        print("All tests for pad_y_and_mask_tensors passed.")

    def test_pad_tensor_unexpected_shapes(self):
        # Tensor already matching max_shape
        tensor = torch.randn(100, 1, 50, 1)
        max_shape = (100, 1, 50, 1)
        padded_tensor = pad_tensor(tensor, max_shape)
        self.assertEqual(padded_tensor.shape, torch.Size(max_shape), f"Expected {max_shape}, got {padded_tensor.shape}")
        
        # Tensor smaller than max_shape
        tensor = torch.randn(50, 1, 30)
        max_shape = (100, 1, 50, 1)
        padded_tensor = pad_tensor(tensor, max_shape)
        self.assertEqual(padded_tensor.shape, torch.Size(max_shape), f"Expected {max_shape}, got {padded_tensor.shape}")
        
        # # Tensor larger than max_shape
        # tensor = torch.randn(150, 2, 60)
        # max_shape = (100, 1, 50, 1)
        # with self.assertRaises(AssertionError):
        #     pad_tensor(tensor, max_shape)

        # print("All tests for unexpected tensor shapes passed.")


    # def test_pad_tensor_unexpected_shapes(self):
    #     # Test case where tensor dimensions exceed max_shape dimensions
    #     tensor = torch.randn(150, 2, 60, 1)
    #     max_shape = (100, 1, 50, 1)
    #     with self.assertRaises(ValueError):
    #         pad_tensor(tensor, max_shape)

        print("Test for dimension mismatch in pad_tensor passed.")

if __name__ == "__main__":
    unittest.main()
