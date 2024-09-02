import torch
import numpy as np
import argparse
import matplotlib
import wandb
import time
import copy
import os
import json
from tueplots import bundles
import matplotlib.pyplot as plt
import constants
import train
import train_transformer
import pred_dists
import visualization as vis
import torch
import torch.nn as nn
import utils

# RNN Cell with decay (to be extended)
class DecayCell(nn.Module):
    def __init__(self, config):
        super(DecayCell, self).__init__()

        self.decay_type = config["decay_type"]
        if config["model"] == "grud_joint":
            # GRU model can be viewed as graph with one node
            self.num_nodes = 1
        else:
            self.num_nodes = config["num_nodes"]
        self.periodic = bool(config["periodic"])

        # Compute number of hidden states needed from W and U
        self.n_states_internal = 3

        if config["decay_type"] == "none":
            if config["model"] == "grud_joint": # Works on all nodes combined
               self.decay_target = torch.zeros(1, config["hidden_dim"],
                       device=config["device"])
            else:
                self.decay_target = torch.zeros(config["num_nodes"],config["hidden_dim"],
                    device=config["device"])
            self.decay_weight = torch.ones(config["hidden_dim"], device=config["device"])
        elif config["decay_type"] == "to_const":
            # Init non-negative
            self.decay_weight = nn.Parameter(torch.rand(config["hidden_dim"],
                device=config["device"]))

            if config["model"] == "grud_joint": # Works on all nodes combined
               self.decay_target = torch.zeros(1, config["hidden_dim"],
                       device=config["device"])
            elif config["node_params"]:
                # Node parameter for decay targets
                self.decay_target = utils.new_param(config["num_nodes"],
                        config["hidden_dim"])
            else:
                self.decay_target = torch.zeros(config["num_nodes"],
                        config["hidden_dim"], device=config["device"])
        elif config["decay_type"] == "dynamic":
            # r, z, \tilde{h} also for drift target
            # Also decay_weight output
            self.n_states_internal += 4
        else:
            assert False, f"Unknown decay type (decay_type): {config['decay_type']}"

        self.DECAY_CAP = 1000 # For numerics

    def compute_inner_states(self, inputs, h_decayed, edge_index):
        raise NotImplementedError()

    def decay_state(self, hidden_state, decay_weight, delta_ts):
        # hidden_state: (BN, d_h)
        # decay_weight: (d_h,) or (BN, d_h)
        # delta_ts: (B, N_t, 1)
        B, N_t, _ = delta_ts.shape
        d_h = hidden_state.shape[-1]

        if self.decay_type == "none":
            # Do not decay state
            return torch.repeat_interleave(hidden_state.unsqueeze(1), N_t,
                    dim=1) # (BN, N_t, d_h)

        if self.periodic:
            # apply time-dependent rotation matrix to pairs of hidden dimensions
            d_h2 = int(d_h/2)

            # In periodic mode half of decay_weight acts as the frequency of rotation
            exp_decay_weight, freq = torch.chunk(decay_weight, 2, dim=-1)
            # each is (d_h/2,) or (BN, d_h/2)

            z1, z2 = torch.chunk(hidden_state, 2, dim=-1) # each is (BN, d_h/2)
            delta_ts = delta_ts.repeat_interleave(self.num_nodes, dim=0) # (BN, N_T, 1)

            freq_rs = freq.view(-1, 1, d_h2) # (BN/1, 1, d_h/2)
            angle = freq_rs*delta_ts # (BN, N_t, d_h/2)
            cos_t = torch.cos(angle) # (BN, N_t, d_h/2)
            sin_t = torch.sin(angle) # (BN, N_t, d_h/2)

            exp_decay_weight_rs = exp_decay_weight.view(-1, 1, d_h2) # (1/BN, 1, d_h/2)
            decay_factor = torch.exp(-1*torch.clamp(delta_ts*exp_decay_weight_rs, min=0.,
                max=self.DECAY_CAP)) # Shape (BN, N_t, d_h/2)

            z1_rs, z2_rs = z1.unsqueeze(1), z2.unsqueeze(1)
            new_z1 = (z1_rs*cos_t - z2_rs*sin_t)*decay_factor # (B, N_t, d_h/2)
            new_z2 = (z1_rs*sin_t + z2_rs*cos_t)*decay_factor # (B, N_t, d_h/2)

            new_dynamic_state = torch.cat((new_z1, new_z2), dim=-1) # (B, N_t, d_h)
        else:
            decay_weight_rs = decay_weight.view(-1,1,d_h) # (BN/1, 1, d_h)

            delta_ts = delta_ts.repeat_interleave(self.num_nodes, dim=0) # (BN, N_T, 1)
            decay_factor = torch.exp(-1*torch.clamp(delta_ts*decay_weight_rs, min=0.,
                max=self.DECAY_CAP)) # Shape (BN, N_t, d_h)

            # hidden-state --> 0 (decaying as decay factor 1 --> 0)
            state_rs = hidden_state.view(-1, self.num_nodes,
                    1, d_h) # (B, N, 1, d_h)
            B = state_rs.shape[0]
            new_dynamic_state = state_rs*decay_factor.view(B, -1,
                    N_t, d_h) # (B, N/1, N_t, d_h)
            new_dynamic_state = new_dynamic_state.view(-1, N_t, d_h) # (BN, N_t, d_h)

        return new_dynamic_state # Shape (BN, N_t, d_h)

    def forward(self, inputs, h_ode, decay_target, decay_weight, delta_ts,
            edge_index, edge_weight):
        # inputs: (B, d_in)
        # h_ode: (B, d_h)
        # decay_target: (B, d_h)
        # decay_weight: (d_h,) or (B, d_h)
        # delta_ts: (B, 1)
        # edge_index: (2, N_edges)
        # edge_weight: (N_edges,)

        h_decayed = self.decay_state(h_ode, decay_weight,
                delta_ts.unsqueeze(1))[:,0]
        Wx, Uh = self.compute_inner_states(inputs, h_decayed+decay_target, edge_index,
                edge_weight)

        W_chunks = Wx.chunk(self.n_states_internal,1)
        U_chunks = Uh.chunk(self.n_states_internal,1)

        Wx_r, Wx_z, Wx_h = W_chunks[:3]
        Uh_r, Uh_z, Uh_h = U_chunks[:3]

        r = torch.sigmoid(Wx_r + Uh_r)
        z = torch.sigmoid(Wx_z + Uh_z)
        h_tilde = torch.tanh(Wx_h + Uh_h*r) # Shape (B, d_h)

        new_h = h_decayed + z*(h_tilde - h_decayed)

        # Compute decay target for time interval
        if self.decay_type == "dynamic":
            # Parameters for decay target parametrization
            Wx_rd, Wx_zd, Wx_hd = W_chunks[3:6]
            Uh_rd, Uh_zd, Uh_hd = U_chunks[3:6]

            rd = torch.sigmoid(Wx_rd + Uh_rd)
            zd = torch.sigmoid(Wx_zd + Uh_zd)
            hd_tilde = torch.tanh(Wx_hd + Uh_hd*rd) # Shape (B, d_h)

            new_decay_target = decay_target + zd*(hd_tilde - decay_target)

            # Decay weight
            Wx_decay_weight = W_chunks[6]
            Uh_decay_weight = U_chunks[6]
            decay_weight = nn.functional.softplus(Wx_decay_weight + Uh_decay_weight)
        else:
            # Compute batch size here (in terms of graphs)
            num_graphs = int(inputs.shape[0] / self.decay_target.shape[0])
            new_decay_target = self.decay_target.repeat(num_graphs, 1)

            decay_weight = self.decay_weight

        new_h_ode = new_h - new_decay_target
        return h_decayed, new_h_ode, new_decay_target, decay_weight


class GRUDecayCell(DecayCell):
    # (Very loosely) Adapted from https://github.com/YuliaRubanova/latent_ode/
    # which was adapted from https://github.com/zhiyongc/GRU-D
    def __init__(self, input_size, config):
        super(GRUDecayCell, self).__init__(config)
        self.W = nn.Linear(input_size, self.n_states_internal*config["hidden_dim"],
                bias=True) # bias vector from here
        self.U = nn.Linear(config["hidden_dim"],
                self.n_states_internal*config["hidden_dim"], bias=False)

    def compute_inner_states(self, inputs, h_decayed, edge_index, edge_weight):
        Wx = self.W(inputs) # Shape (B, n_states_internal*d_h)
        Uh = self.U(h_decayed) # Shape (B, n_states_internal*d_h)

        return Wx, Uh


class GRUModel(nn.Module):
    # Handles all nodes together in a single GRU-unit
    def __init__(self, config):
        super(GRUModel, self).__init__()

        self.time_input = bool(config["time_input"])
        self.mask_input = bool(config["mask_input"])
        self.has_features = config["has_features"]
        self.max_pred = config["max_pred"]

        output_dim = self.compute_output_dim(config)

        self.gru_cells = self.create_cells(config)

        if config["learn_init_state"]:
            self.init_state_param = utils.new_param(config["gru_layers"],
                    config["hidden_dim"])
        else:
            self.init_state_param = torch.zeros(config["gru_layers"],
                    config["hidden_dim"], device=config["device"])

        first_post_dim = self.compute_pred_input_dim(config)
        if config["n_fc"] == 1:
            fc_layers = [nn.Linear(first_post_dim, output_dim)]
        else:
            fc_layers = []
            for layer_i in range(config["n_fc"]-1):
                fc_layers.append(nn.Linear(first_post_dim if (layer_i == 0)
                    else config["hidden_dim"], config["hidden_dim"]))
                fc_layers.append(nn.ReLU())

            fc_layers.append(nn.Linear(config["hidden_dim"], output_dim))

        self.post_gru_layers = nn.Sequential(*fc_layers)

        self.y_shape = (config["time_steps"], -1, config["num_nodes"]*config["y_dim"])
        self.delta_t_shape = (config["time_steps"], -1, config["num_nodes"])
        # Return shape
        self.pred_shape = (config["time_steps"], -1, config["max_pred"],
                config["y_dim"],config["param_dim"])
        self.f_shape = (config["time_steps"], -1,
                config["num_nodes"]*config["feature_dim"])

        self.init_decay_weight = torch.zeros(config["hidden_dim"],
                device=config["device"])

    def create_cells(self, config):
        input_dim = self.compute_gru_input_dim(config)

        return nn.ModuleList([
                GRUDecayCell(input_dim if layer_i==0 else config["hidden_dim"], config)
            for layer_i in range(config["gru_layers"])])

    def compute_gru_input_dim(self, config):
        # Compute input dimension at each timestep
        return config["num_nodes"]*(config["y_dim"] + config["feature_dim"] +
                int(self.mask_input)+ int(self.time_input))  # Add N if time/mask input

    def compute_pred_input_dim(self, config):
        return config["hidden_dim"] + config["num_nodes"]*config["feature_dim"] +\
                int(self.time_input) # Add 1 if delta_t input

    def compute_output_dim(self, config):
        # Compute output dimension at each timestep
        return config["num_nodes"]*config["param_dim"]*config["y_dim"]

    def get_init_states(self, num_graphs):
        return self.init_state_param.unsqueeze(1).repeat(
                1, num_graphs, 1) # Output shape (n_gru_layers, B, d_h)

    def forward(self, batch):
        # Batch is ptg-Batch: Data(
        #  y: (BN, N_T, 1)
        #  t: (B, N_T)
        #  delta_t: (BN, N_T)
        #  mask: (BN, N_T)
        # )

        edge_weight = batch.edge_attr[:,0] # Shape (N_edges,)

        input_y_full = batch.y.transpose(0,1) # Shape (N_T, B*N, d_y)
        input_y_reshaped = input_y_full.reshape(self.y_shape) # (N_T, B, N*d_y)

        delta_time = batch.delta_t.transpose(0,1).unsqueeze(-1) # Shape (N_T, B*N, 1)

        all_dts = batch.t.unsqueeze(1) - batch.t.unsqueeze(2) # (B, N_T, N_T)
        # Index [:, i, j] is (t_j - t_i), time from t_i to t_j
        off_diags = [torch.diagonal(all_dts, offset=offset, dim1=1, dim2=2).t()
                for offset in range(self.max_pred+1)]
        # List of length max_preds, each entry is tensor: (diag_length, B)
        padded_off_diags = torch.nn.utils.rnn.pad_sequence(off_diags,
                batch_first=False) # (N_T, max_pred+1, B)

        pred_delta_times = padded_off_diags[:,1:].transpose(1,2) # (N_T, B, max_pred)
        # Index [i, :, j] is (t_(i+j) - t_i), time from t_i to t_(i+j)

        all_delta_ts = utils.t_to_delta_t(batch.t).transpose(
                0,1).unsqueeze(2) # (N_T, B, 1)

        # Only for mask input here
        obs_mask = batch.mask.transpose(0,1).view(
                *self.delta_t_shape) # Shape (N_T, B, N)

        # List with all tensors for input
        gru_input_tensors = [input_y_reshaped,] # input to gru update

        if self.has_features:
            input_f_full = batch.features.transpose(0,1) # Shape (N_T, B*N, d_f)
            input_f_reshaped = input_f_full.reshape(self.f_shape) # (N_T, B, N*d_f)
            gru_input_tensors.append(input_f_reshaped)
            fc_feature_input = input_f_reshaped.transpose(0,1) # (B, N_T, N*d_f)

            # Pad feature input for last time steps
            feature_padding = torch.zeros_like(fc_feature_input)[:,:self.max_pred,:]
            fc_feature_input = torch.cat((fc_feature_input, feature_padding), dim=1)
            # (B, N_T+max_pred, N*d_f)

        if self.time_input:
            delta_time_inputs = batch.delta_t.transpose(0,1).view(
                self.delta_t_shape) # (N_T, B, N)

            # Concatenated delta_t to input
            gru_input_tensors.append(delta_time_inputs)

        if self.mask_input:
            # Concatenated mask to input (does not always make sense)
            gru_input_tensors.append(obs_mask)
            # Mask should not be in fc_input, we don't
            # know what will be observed when predicting

        init_states = self.get_init_states(batch.num_graphs)

        gru_input = torch.cat(gru_input_tensors, dim=-1) # (N_T, B, d_gru_input)
        for layer_i, (gru_cell, init_state) in enumerate(
                zip(self.gru_cells, init_states)):
            h_ode = torch.zeros_like(init_state)
            decay_target = init_state # Init decaying from and to initial state
            decay_weight = self.init_decay_weight # dummmy (does not matter)

            step_preds = [] # predictions from each step
            hidden_states = [] # New states after observation
            for t_i, (input_slice,\
                delta_time_slice,\
                pred_delta_time_slice)\
            in enumerate(zip(
                gru_input,\
                all_delta_ts,\
                pred_delta_times\
            )):
                # input_slice: (B, d_gru_input)
                # delta_time_slice: (B,1)
                # pred_delta_time_slice: (B, max_pred)

                # STATE UPDATE
                decayed_states, new_h_ode, new_decay_target, new_decay_weight =\
                    gru_cell(input_slice, h_ode, decay_target, decay_weight,
                        delta_time_slice, batch.edge_index, edge_weight)

                # Update for all nodes
                h_ode = new_h_ode
                decay_target = new_decay_target
                decay_weight = new_decay_weight

                # Hidden state is sum of ODE-controlled state and decay target
                hidden_state = h_ode + decay_target
                hidden_states.append(hidden_state)

                # PREDICTION
                if layer_i == (len(self.gru_cells)-1):
                    # Decay to all future time points for prediction
                    decayed_pred_h_ode = gru_cell.decay_state(h_ode,
                            decay_weight, pred_delta_time_slice.unsqueeze(-1))
                    decayed_pred_states = decayed_pred_h_ode + decay_target.unsqueeze(1)
                    # decayed_pred_states is (B, max_pred, d_h)

                    # Perform prediction
                    pred_input_list = [decayed_pred_states]
                    if self.time_input:
                        # Time from now until prediction
                        time_input = pred_delta_time_slice.unsqueeze(
                                -1) # (B, max_pred, 1)
                        pred_input_list.append(time_input)

                    if self.has_features:
                        features_for_time = fc_feature_input[
                                :,(t_i+1):(t_i+1+self.max_pred)] # (B, max_pred, N*d_f)
                        pred_input_list.append(features_for_time)

                    pred_input = torch.cat(pred_input_list,
                            dim=-1) # (B, max_pred, d_h+d_aux)

                    step_prediction = self.post_gru_layers(
                            pred_input) # (B, max_pred, N*d_out)
                    step_preds.append(step_prediction)

            gru_input = hidden_states

        predictions = torch.cat(step_preds, dim=0) # (N_T, BN, max_pred, N*d_out)
        predictions_reshaped = predictions.view(
                *self.pred_shape) # (N_T, BN, max_pred, d_y, d_param)
        return predictions_reshaped, pred_delta_times


class GRUNodeModel(GRUModel):
    # Handles each node independently with GRU-units
    def __init__(self, config):
        super(GRUNodeModel, self).__init__(config)

        self.num_nodes = config["num_nodes"]
        self.y_shape = (config["time_steps"], -1, config["y_dim"])
        self.f_shape = (config["time_steps"], -1, config["feature_dim"])
        self.state_updates = config["state_updates"]
        assert config["state_updates"] in ("all", "obs", "hop"), (
                f"Unknown state update: {config['state_updates']}")

        # If node-specific initial states should be used
        self.node_init_states = (config["node_params"] and config["learn_init_state"])
        if self.node_init_states:
            # Override initial GRU-states
            self.init_state_param = utils.new_param(config["gru_layers"],
                    config["num_nodes"], config["hidden_dim"])

    def get_init_states(self, num_graphs):
        if self.node_init_states:
            return self.init_state_param.repeat(
                1, num_graphs, 1) # Output shape (n_gru_layers, B*N, d_h)
        else:
            return self.init_state_param.unsqueeze(1).repeat(
                1, self.num_nodes*num_graphs, 1) # Output shape (n_gru_layers, B*N, d_h)

    def compute_gru_input_dim(self, config):
        # Compute input dimension at each timestep
        return config["y_dim"] + config["feature_dim"] +\
            int(self.time_input) + int(self.mask_input) # Add one if delta_t/mask input

    def compute_pred_input_dim(self, config):
        return config["hidden_dim"] + config["feature_dim"] +\
                int(self.time_input) # Add one if delta_t input

    def compute_output_dim(self, config):
        # Compute output dimension at each timestep
        return config["param_dim"]*config["y_dim"]

    def compute_predictions(self, pred_input, edge_index, edge_weight):
        # pred_input: (N_T, B*N, N_T, pred_input_dim)
        return self.post_gru_layers(pred_input) # Shape (N_T, B*N, N_T, d_out)

    def forward(self, batch):
        # Batch is ptg-Batch: Data(
        #  y: (BN, N_T, 1)
        #  t: (B, N_T)
        #  delta_t: (BN, N_T)
        #  mask: (BN, N_T)
        # )

        edge_weight = batch.edge_attr[:,0] # Shape (N_edges,)

        input_y_full = batch.y.transpose(0,1) # Shape (N_T, B*N, d_y)
        input_y_reshaped = input_y_full.reshape(self.y_shape) # (N_T, B*N, d_y)

        delta_time = batch.delta_t.transpose(0,1).unsqueeze(-1) # Shape (N_T, B*N, 1)

        all_dts = batch.t.unsqueeze(1) - batch.t.unsqueeze(2) # (B, N_T, N_T)
        # Index [:, i, j] is (t_j - t_i), time from t_i to t_j
        off_diags = [torch.diagonal(all_dts, offset=offset, dim1=1, dim2=2).t()
                for offset in range(self.max_pred+1)]
        # List of length max_preds, each entry is tensor: (diag_length, B)
        padded_off_diags = torch.nn.utils.rnn.pad_sequence(off_diags,
                batch_first=False) # (N_T, max_pred+1, B)

        pred_delta_times = padded_off_diags[:,1:].transpose(1,2) # (N_T, B, max_pred)
        # Index [i, :, j] is (t_(i+j) - t_i), time from t_i to t_(i+j)

        all_delta_ts = utils.t_to_delta_t(batch.t).transpose(
                0,1).unsqueeze(2) # (N_T, B, 1)
        dt_node_obs = batch.update_delta_t.transpose(0,1).unsqueeze(2) # (N_T, BN, 1)

        obs_mask = batch.mask.transpose(0,1).unsqueeze(-1) # Shape (N_T, B*N, 1)

        if self.state_updates == "hop":
            update_mask = batch.hop_mask.transpose(0,1).unsqueeze(-1) # (N_T, B*N, 1)
        else:
            # "obs" (or "all", but then unused)
            update_mask = obs_mask

        # List with all tensors for input
        gru_input_tensors = [input_y_reshaped,] # input to gru update

        if self.has_features:
            input_f_full = batch.features.transpose(0,1) # Shape (N_T, B*N, d_f)
            input_f_reshaped = input_f_full.reshape(self.f_shape) # (N_T, BN, d_f)
            gru_input_tensors.append(input_f_reshaped)
            fc_feature_input = input_f_reshaped.transpose(0,1) # (BN, N_T, d_f)

            # Pad feature input for last time steps
            feature_padding = torch.zeros_like(fc_feature_input)[:,:self.max_pred,:]
            fc_feature_input = torch.cat((fc_feature_input, feature_padding), dim=1)
            # (BN, N_T+max_pred, d_f)

        if self.time_input:
            gru_input_tensors.append(delta_time)

        if self.mask_input:
            # Concatenated mask to input (does not always make sense)
            gru_input_tensors.append(obs_mask)
            # Mask should not be in fc_input, we don't
            # know what will be observed when predicting

        init_states = self.get_init_states(batch.num_graphs)

        gru_input = torch.cat(gru_input_tensors, dim=-1)
        for layer_i, (gru_cell, init_state) in enumerate(
                zip(self.gru_cells, init_states)):
            h_ode = torch.zeros_like(init_state)
            decay_target = init_state # Init decaying from and to initial state
            decay_weight = self.init_decay_weight # dummmy (does not matter)

            step_preds = [] # predictions from each step
            hidden_states = [] # New states after observation
            for t_i, (input_slice,\
                delta_time_slice,\
                update_mask_slice,\
                pred_delta_time_slice,\
                dt_node_obs_slice,)\
            in enumerate(zip(
                gru_input,\
                all_delta_ts,\
                update_mask,\
                pred_delta_times,\
                dt_node_obs
            )):
                # input_slice: (BN, d_gru_input)
                # delta_time_slice: (B,1)
                # update_mask_slice: (BN,1)
                # pred_delta_time_slice: (B, max_pred)
                # dt_node_obs_slice: (BN, 1)

                # STATE UPDATE
                decayed_states, new_h_ode, new_decay_target, new_decay_weight =\
                    gru_cell(input_slice, h_ode, decay_target, decay_weight,
                        delta_time_slice, batch.edge_index, edge_weight)

                if self.state_updates == "all":
                    # Update for all nodes
                    h_ode = new_h_ode
                    decay_target = new_decay_target
                    decay_weight = new_decay_weight
                else:
                    # GRU update for observed nodes, decay others
                    h_ode = update_mask_slice*new_h_ode +\
                        (1. - update_mask_slice)*decayed_states
                    decay_target = update_mask_slice*new_decay_target +\
                        (1. - update_mask_slice)*decay_target
                    decay_weight = update_mask_slice*new_decay_weight +\
                        (1. - update_mask_slice)*decay_weight

                # Hidden state is sum of ODE-controlled state and decay target
                hidden_state = h_ode + decay_target
                hidden_states.append(hidden_state)

                # PREDICTION
                if layer_i == (len(self.gru_cells)-1):
                    # Decay to all future time points for prediction
                    decayed_pred_h_ode = gru_cell.decay_state(h_ode,
                            decay_weight, pred_delta_time_slice.unsqueeze(-1))
                    decayed_pred_states = decayed_pred_h_ode + decay_target.unsqueeze(1)
                    # decayed_pred_states is (BN, max_pred, d_h)

                    # Perform prediction
                    pred_input_list = [decayed_pred_states]
                    if self.time_input:
                        # Note: Time since last node obs for each prediction is
                        # sum of dt_node_obs and pred_delta_tim
                        # dt_node_obs_slice: (BN, 1)
                        # pred_delta_time_slice: (B, max_pred)

                        # 0 time since update for nodes updated at this time
                        node_time_since_up = dt_node_obs_slice*(1. - update_mask_slice)

                        time_input = node_time_since_up.view(-1, self.num_nodes, 1) +\
                            pred_delta_time_slice.unsqueeze(1) # (B, N, max_pred)
                        BN = dt_node_obs_slice.shape[0]
                        time_input = time_input.view(BN, -1, 1) # (BN, max_pred, 1)
                        pred_input_list.append(time_input)

                    if self.has_features:
                        features_for_time = fc_feature_input[
                                :,(t_i+1):(t_i+1+self.max_pred)] # (BN, max_pred, d_f)
                        pred_input_list.append(features_for_time)

                    pred_input = torch.cat(pred_input_list,
                            dim=-1) # (BN, max_pred, d_h+d_aux)

                    step_prediction = self.compute_predictions(pred_input,
                            batch.edge_index, edge_weight) # (BN, max_pred, d_out)
                    step_preds.append(step_prediction)

            gru_input = hidden_states

        predictions = torch.cat(step_preds, dim=0) # (N_T, BN, max_pred, d_out)
        predictions_reshaped = predictions.view(
                *self.pred_shape) # (N_T, BN, max_pred, d_y, d_param)
        return predictions_reshaped, pred_delta_times


class GRUGraphCell(DecayCell):
    def __init__(self, input_size, config):
        super(GRUGraphCell, self).__init__(config)

        hidden_dim = config["hidden_dim"]
        out_dim = self.n_states_internal*hidden_dim
        self.input_gnn = utils.build_gnn_seq(config["gru_gnn"], input_size, hidden_dim,
                out_dim, config["gnn_type"])
        self.state_gnn = utils.build_gnn_seq(config["gru_gnn"], hidden_dim, hidden_dim,
                out_dim, config["gnn_type"])

    def compute_inner_states(self, inputs, h_decayed, edge_index,
            edge_weight):
        Wx = self.input_gnn(inputs, edge_index,
                edge_weight=edge_weight) # Shape (B, n_states_internal*d_h)
        Uh = self.state_gnn(h_decayed, edge_index,
                edge_weight=edge_weight) # Shape (B, n_states_internal*d_h)

        return Wx, Uh


class GRUGraphModel(GRUNodeModel):
    def __init__(self, config):
        self.pred_gnn = bool(config["pred_gnn"])

        super(GRUGraphModel, self).__init__(config)

        if self.pred_gnn:
            pred_gnn_input_dim = super(GRUGraphModel, self).compute_pred_input_dim(config)

            # instantiate a GNN to use for predictions, output dim here is hidden dim
            self.pred_gnn_model = utils.build_gnn_seq(config["pred_gnn"],
                    pred_gnn_input_dim, config["hidden_dim"], config["hidden_dim"],
                    config["gnn_type"])

    def create_cells(self, config):
        if config["gru_gnn"]:
            input_dim = self.compute_gru_input_dim(config)

            return nn.ModuleList([
                    GRUGraphCell(input_dim if layer_i==0 else config["hidden_dim"], config)
                for layer_i in range(config["gru_layers"])])
        else:
            # No GNN for GRU, use normal (decaying) GRU-cell
            return super(GRUGraphModel, self).create_cells(config)

    def compute_pred_input_dim(self, config):
        if self.pred_gnn:
            return config["hidden_dim"]
        else:
            return super(GRUGraphModel, self).compute_pred_input_dim(config)

    def compute_predictions(self, pred_input, edge_index, edge_weight):
        # pred_input: (BN, N_T, pred_input_dim)

        pred_input = pred_input.transpose(0,1)

        if self.pred_gnn:
            post_gnn = self.pred_gnn_model(pred_input, edge_index,
                        edge_weight) # (N_T, N_T, B*N, hidden_dim)

            fc_input = nn.functional.relu(post_gnn) # Activation in-between
        else:
            fc_input = pred_input

        return self.post_gru_layers(fc_input).transpose(0,1) # Shape (B*N, N_T, d_out)


class TransformerForecaster(nn.Module):
    def __init__(self, config):
        super(TransformerForecaster, self).__init__()

        self.num_nodes = config["num_nodes"]
        self.pred_len = config["max_pred"]

        self.pos_encode_dim = config["hidden_dim"]
        self.input_encoder = nn.Linear(2, config["hidden_dim"])
        self.trans_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(config["hidden_dim"], 4, config["hidden_dim"],
                    batch_first=True), config["gru_layers"])

        self.trans_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(config["hidden_dim"], 4, config["hidden_dim"],
                    batch_first=True), config["gru_layers"])

        self.pred_model = nn.Linear(config["hidden_dim"], 1)

        self.prior_info = nn.Parameter(torch.randn(config["hidden_dim"]))

    def encode_time(self, t):
        # t: (B, N_t)
        i = torch.arange(self.pos_encode_dim // 2, device=t.device)
        denom = 0.1**(2*i / self.pos_encode_dim) # (self.pos_encode_dim/2,)
        f = t.unsqueeze(-1) / denom

        encoding = torch.cat((torch.sin(f), torch.cos(f)),
                dim=-1) # (B, N_T, pos_encoding_dim)
        return encoding

    def forward(self, batch, cond_length):
        # Batch is ptg-Batch: Data(
        #  y: (BN, N_T, 1)
        #  t: (B, N_T)
        #  delta_t: (BN, N_T)
        #  mask: (BN, N_T)
        # )

        pos_enc = self.encode_time(batch.t) #(B, N_T, d_h)
        pos_enc_repeated = pos_enc.repeat_interleave(self.num_nodes,
                dim=0) # (BN, N_T, d_h)
        enc_input = torch.cat((batch.y, batch.mask.unsqueeze(-1)),
                dim=-1) # (B, N_T, d_h)

        # Only encode conditioning length
        trans_input = self.input_encoder(enc_input[:,:cond_length]) +\
            pos_enc_repeated[:,:cond_length] # (B, N_T', d_h)

        # Treat unobserved times as padding
        enc_mask = batch.mask.to(bool).logical_not()[
                :,:cond_length] # True when batch.mask is 0 (unobs.) (B, N_T')

        # Add on static prior info representation
        # (Fixes cases where nothing obs. in encoding seq)
        prior_info_rs = self.prior_info.view(1,1,self.prior_info.shape[0]
                ).repeat_interleave(trans_input.shape[0], dim=0)
        trans_input = torch.cat((trans_input, prior_info_rs), dim=1) # (B, N_T'+1, d_h)
        extra_mask = torch.zeros((enc_mask.shape[0],1),
                device=enc_mask.device).to(bool) # (B,1)
        enc_mask = torch.cat((enc_mask, extra_mask), dim=1) #(B, N_T'+1)

        encoded_rep = self.trans_encoder(trans_input,
                src_key_padding_mask=enc_mask) # (B, N_T', d_h)

        # Input to decoder is only time encoding
        dec_input = pos_enc_repeated[:,cond_length:(cond_length+self.pred_len)]
        decoded_rep = self.trans_decoder(dec_input, encoded_rep,
                memory_key_padding_mask=enc_mask) # (B, max_pred, d_h)

        pred = self.pred_model(decoded_rep) # (B, max_pred, 1)

        # Pad in case of short output len
        actual_pred_len = pred.shape[1]
        if actual_pred_len < self.pred_len:
            pred_padding = torch.zeros(pred.shape[0],
                    (self.pred_len - actual_pred_len), 1, device=pred.device)
            pred = torch.cat((pred, pred_padding), dim=1)

        return pred


class TransformerJointForecaster(nn.Module):
    def __init__(self, config):
        super(TransformerJointForecaster, self).__init__()

        self.num_nodes = config["num_nodes"]
        self.pred_len = config["max_pred"]

        self.pos_encode_dim = config["hidden_dim"]
        self.input_encoder = nn.Linear(2*self.num_nodes, config["hidden_dim"])
        self.trans_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(config["hidden_dim"], 4, config["hidden_dim"],
                    batch_first=True), config["gru_layers"])

        self.trans_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(config["hidden_dim"], 4, config["hidden_dim"],
                    batch_first=True), config["gru_layers"])

        self.pred_model = nn.Linear(config["hidden_dim"], self.num_nodes)

        self.prior_info = nn.Parameter(torch.randn(config["hidden_dim"]))

    def encode_time(self, t):
        # t: (B, N_t)
        i = torch.arange(self.pos_encode_dim // 2, device=t.device)
        denom = 0.1**(2*i / self.pos_encode_dim) # (self.pos_encode_dim/2,)
        f = t.unsqueeze(-1) / denom

        encoding = torch.cat((torch.sin(f), torch.cos(f)),
                dim=-1) # (B, N_T, pos_encoding_dim)
        return encoding

    def forward(self, batch, cond_length):
        # Batch is ptg-Batch: Data(
        #  y: (BN, N_T, 1)
        #  t: (B, N_T)
        #  delta_t: (BN, N_T)
        #  mask: (BN, N_T)
        # )

        pos_enc = self.encode_time(batch.t) #(B, N_T, d_h)

        N_T = batch.t.shape[1]
        y = batch.y.view(-1, self.num_nodes, N_T).transpose(1,2) # (B, N_T, N)
        mask_input = batch.mask.view(-1, self.num_nodes,
                N_T).transpose(1,2) # (B, N_T, N)
        enc_input = torch.cat((y, mask_input),
                dim=-1) # (B, N_T, 2N)

        # Only encode conditioning length
        trans_input = self.input_encoder(enc_input[:,:cond_length]) +\
            pos_enc[:,:cond_length] # (B, N_T', d_h)

        # Add on static prior info representation
        # (Fixes cases where nothing obs. in encoding seq)
        prior_info_rs = self.prior_info.view(1,1,self.prior_info.shape[0]
                ).repeat_interleave(trans_input.shape[0], dim=0)
        trans_input = torch.cat((trans_input, prior_info_rs), dim=1) # (B, N_T'+1, d_h)

        encoded_rep = self.trans_encoder(trans_input) # (B, N_T', d_h)

        # Input to decoder is only time encoding
        dec_input = pos_enc[:,cond_length:(cond_length+self.pred_len)] # (B, max_pred)
        decoded_rep = self.trans_decoder(dec_input, encoded_rep) # (B, max_pred, d_h)

        pred = self.pred_model(decoded_rep) # (B, max_pred, N)

        # Pad in case of short output len
        actual_pred_len = pred.shape[1]
        if actual_pred_len < self.pred_len:
            pred_padding = torch.zeros(pred.shape[0],
                    (self.pred_len - actual_pred_len), self.num_nodes,
                    device=pred.device)
            pred = torch.cat((pred, pred_padding), dim=1) # (B, max_pred, N)

        # Reshape pred
        pred = pred.transpose(1,2).reshape(-1, self.pred_len, 1) # (BN, max_pred, 1)
        return pred


def get_config():
    parser = argparse.ArgumentParser(description='Train Models')
    # If config file should be used
    parser.add_argument("--config", type=str, help="Config file to read run config from")

    # General
    parser.add_argument("--model", type=str, default="tgnn4i",
            help="Which dataset to use")
    parser.add_argument("--dataset", type=str, default="la_node_0.25",
            help="Which dataset to use")
    parser.add_argument("--seed", type=int, default=42,
            help="Seed for random number generator")
    parser.add_argument("--optimizer", type=str, default="adam",
            help="Optimizer to use for training")
    parser.add_argument("--init_points", type=int, default=5,
            help="Number of points to observe before prediction start")
    parser.add_argument("--test", type=int, default=0,
            help="Also evaluate on test set after training is done")
    parser.add_argument("--use_features", type=int, default=1,
            help="If additional input features should be used")
    parser.add_argument("--load", type=str,
            help="Load model parameters from path")

    # Model Architecture
    parser.add_argument("--gru_layers", type=int, default=1,
            help="Layers of GRU units")
    parser.add_argument("--decay_type", type=str, default="dynamic",
            help="Parametrization of GRU decay to use (none/to_const/dynamic)")
    parser.add_argument("--periodic", type=int, default=0,
            help="If latent state dynamics should include periodic component")
    parser.add_argument("--time_input", type=int, default=1,
            help="Concatenate time (delta_t) to the input at each timestep")
    parser.add_argument("--mask_input", type=int, default=1,
            help="Concatenate the observation mask as input")
    parser.add_argument("--hidden_dim", type=int, default=32,
            help="Dimensionality of hidden state in GRU units (latent node state))")
    parser.add_argument("--n_fc", type=int, default=2,
            help="Number of fully connected layers after GRU units")
    parser.add_argument("--pred_gnn", type=int, default=1,
            help="Number of GNN-layers to use in predictive part of model")
    parser.add_argument("--gru_gnn", type=int, default=1,
            help="Number of GNN layers used for GRU-cells")
    parser.add_argument("--gnn_type", type=str, default="graphconv",
            help="Type of GNN-layers to use")
    parser.add_argument("--node_params", type=int, default=1,
            help="Use node-specific parameters for initial state and decay target")
    parser.add_argument("--learn_init_state", type=int, default=1,
            help="If the initial state of GRU-units should be learned (otherwise 0)")

    # Training
    parser.add_argument("--epochs", type=int,
            help="How many epochs to train for", default=5)
    parser.add_argument("--val_interval", type=int, default=1,
            help="Evaluate model every val_interval:th epoch")
    parser.add_argument("--patience", type=int, default=20,
            help="How many evaluations to wait for improvement in val loss")
    parser.add_argument("--pred_dist", type=str, default="gauss_fixed",
            help="Predictive distribution")
    parser.add_argument("--lr", type=float,
            help="Learning rate", default=1e-3)
    parser.add_argument("--l2_reg", type=float,
            help="L2-regularization coefficient", default=0.)
    parser.add_argument("--batch_size", type=int,
            help="Batch size", default=32)
    parser.add_argument("--state_updates", type=str, default="obs",
            help="When the node state should be updated (all/obs/hop)")
    parser.add_argument("--loss_weighting", type=str, default="exp,0.04",
            help="Function to weight loss with, given as: name,param1,...,paramK")
    parser.add_argument("--max_pred", type=int, default=10,
            help="Maximum number of time indices forward to predict")

    # Plotting
    parser.add_argument("--plot_pred", type=int, default=3,
            help="Number of prediction plots to make")
    parser.add_argument("--max_nodes_plot", type=int, default=3,
            help="Maximum number of nodes to plot predictions for")
    parser.add_argument("--save_pdf", type=int, default=0,
            help="If pdf:s should be generated for plots (NOTE: Requires much space)")

    args = parser.parse_args()
    config = vars(args)

    # Read additional config from file
    if args.config:
        assert os.path.exists(args.config), "No config file: {}".format(args.config)
        with open(args.config) as json_file:
            config_from_file = json.load(json_file)

        # Make sure all options in config file also exist in argparse config.
        # Avoids choosing wrong parameters because of typos etc.
        unknown_options = set(config_from_file.keys()).difference(set(config.keys()))
        unknown_error = "\n".join(["Unknown option in config file: {}".format(opt)
            for opt in unknown_options])
        assert (not unknown_options), unknown_error

        config.update(config_from_file)

    # Some asserts
    assert config["model"] in MODELS, f"Unknown model: {config['model']}"
    assert config["optimizer"] in constants.OPTIMIZERS, (
            f"Unknown optimizer: {config['optimizer']}")
    assert config["pred_dist"] in pred_dists.DISTS, (
            f"Unknown predictive distribution: {config['pred_dist']}")
    assert config["gnn_type"] in constants.GNN_LAYERS, (
            f"Unknown gnn_type: {config['gnn_type']}")
    assert config["init_points"] > 0, "Need to have positive number of init points"
    assert (not bool(config["periodic"])) or (config["hidden_dim"] % 2 == 0), (
            "hidden_dim must be even when using periodic latent dynamics")

    if config["plot_pred"] > config["batch_size"]:
        print(f"Warning: Can only make {config['batch_size']} plots")
        config["plot_pred"] = config["batch_size"]

    return config


def main():
    config = get_config()

    # Set all random seeds
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # For reproducability on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    # Load data
    # torch.multiprocessing.set_sharing_strategy('file_system') # Fix for num_workers > 0
    train_loader, val_loader, test_loader = utils.load_temporal_graph_data(
            config["dataset"], config["batch_size"],
            compute_hop_mask=(config["state_updates"] == "hop"), L_hop=config["gru_gnn"])

    # Init wandb
    wandb_name = f"{config['dataset']}_{config['model']}_{time.strftime('%H-%M-%S')}"
    wandb.init(project=constants.WANDB_PROJECT, config=config, name=wandb_name)
    # Additional config needed for some model setup (need/should not be logged to wandb)
    config["num_nodes"] = train_loader.dataset[0].num_nodes
    try:
        config["time_steps"] = train_loader.dataset[0].t.shape[1]
    except:
        import pdb; pdb.set_trace()
    config["device"] = device
    config["y_dim"] = train_loader.dataset[0].y.shape[-1]

    config["has_features"] = hasattr(train_loader.dataset[0], "features") and\
        bool(config["use_features"])
    if config["has_features"]:
        config["feature_dim"] = train_loader.dataset[0].features.shape[-1]
    else:
        config["feature_dim"] = 0

    # param_dim is number of parameters in predictive distribution
    pred_dist, config["param_dim"] = pred_dists.DISTS[config["pred_dist"]]

    # Parse loss weighting function
    loss_weight_func = utils.parse_loss_weight(config["loss_weighting"])

    # Create model, optimizer
    model = MODELS[config["model"]](config).to(device)
    if config["load"]:
        model.load_state_dict(torch.load(config["load"], map_location=device))
        print(f"Parameters loaded from: {config['load']}")
    opt = constants.OPTIMIZERS[config["optimizer"]](model.parameters(), lr=config["lr"],
            weight_decay=config["l2_reg"])

    is_transformer = config["model"].startswith("transformer")
    if is_transformer:
        train_epoch = train_transformer.train_epoch
        val_epoch = train_transformer.val_epoch
        test_epoch = train_transformer.test_epoch
    else:
        train_epoch = train.train_epoch
        val_epoch = train.val_epoch
        test_epoch = train.val_epoch # Note: Same function, but test data should be used

    # Train model
    best_val_loss = np.inf
    best_val_metrics = None
    best_val_epoch = -1 # Index of the best epoch
    best_params = None

    model.train()
    for epoch_i in range(1, config["epochs"]+1):
        epoch_train_loss = train_epoch(model, train_loader, opt, pred_dist,
                config, loss_weight_func)

        if (epoch_i % config["val_interval"]== 0):
            # Validate, evaluate
            with torch.no_grad():
                epoch_val_metrics = val_epoch(model, val_loader, pred_dist,
                        loss_weight_func, config)

            log_metrics = {"train_loss": epoch_train_loss}
            log_metrics.update({f"val_{metric}": val for
                metric, val in epoch_val_metrics.items()})

            epoch_val_loss = log_metrics["val_wmse"] # Use wmse as main metric
            epoch_val_mse = log_metrics["val_mse"]

            print(f"Epoch {epoch_i}:\t train_loss: {epoch_train_loss:.6f} "\
                    f"\tval_wmse: {epoch_val_loss:.6f} \tval_mse: {epoch_val_mse:.6f}")

            wandb.log(log_metrics, step=epoch_i, commit=True)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_val_metrics = log_metrics
                best_val_epoch = epoch_i
                best_params = copy.deepcopy(model.state_dict())

            if (epoch_i - best_val_epoch)/config["val_interval"] >= config["patience"]:
                # No improvement, end training
                print("Val loss no longer improving, stopping training.")
                break

    # Save things
    param_save_path = os.path.join(wandb.run.dir, constants.PARAM_FILE_NAME)
    torch.save(best_params, param_save_path)

    # Restore parameters and plot
    model.load_state_dict(best_params)
    if not is_transformer:
        with torch.no_grad(), plt.rc_context(bundles.aistats2023()):
            val_pred_plots = vis.plot_step_prediction(model, val_loader, config["plot_pred"],
                    pred_dist, config)
            for fig_i, fig in enumerate(val_pred_plots):
                if config["save_pdf"]:
                    save_path = os.path.join(wandb.run.dir, f"val_pred_{fig_i}.pdf")
                    fig.savefig(save_path)

                # wandb.log({"val_pred": wandb.Image(fig)})

            all_pred_plot, bin_errors, unique_dts, unique_counts = vis.plot_all_predictions(
                    model, val_loader, pred_dist, loss_weight_func, config)
            if config["save_pdf"]:
                save_path = os.path.join(wandb.run.dir, f"val_error.pdf")
                all_pred_plot.savefig(save_path)
        #     wandb.log({"val_error": wandb.Image(all_pred_plot)})

    # Wandb summary
    del best_val_metrics["train_loss"] # Exclude this one from summary update
    for metric, val in best_val_metrics.items():
        wandb.run.summary[metric] = val
    if config["decay_type"] == "to_const":
        # Decay parameters histogram
        with torch.no_grad():
            for cell_id, gru_cell in enumerate(model.gru_cells):
                wandb.run.summary.update({f"GRU_{cell_id}_decay":
                    wandb.Histogram(gru_cell.decay_weight.cpu())})

    # (Optionally) Evaluate on test set
    if config["test"]:
        with torch.no_grad(), plt.rc_context(bundles.aistats2023()):
            test_metrics = test_epoch(model, test_loader, pred_dist,
                    loss_weight_func, config)
            test_metric_dict = {f"test_{name}": val
                    for name, val in test_metrics.items()}
            wandb.run.summary.update(test_metric_dict)

            print("Test set evaluation:")
            for name, val in test_metric_dict.items():
                print(f"{name}:\t {val}")

            if not is_transformer:
                # Compute test errors at different delta-ts
                all_pred_plot, bin_errors, unique_dts, unique_counts =\
                    vis.plot_all_predictions(
                        model, val_loader, pred_dist, loss_weight_func, config)
                if config["save_pdf"]:
                    save_path = os.path.join(wandb.run.dir, f"test_error.pdf")
                    all_pred_plot.savefig(save_path)
                # wandb.log({"test_error": wandb.Image(all_pred_plot)})

                # Save binned errors
                np.save(os.path.join(wandb.run.dir, "test_bin_errors.npy"), bin_errors)
                np.save(os.path.join(wandb.run.dir, "test_bin_dt.npy"), unique_dts)
                np.save(os.path.join(wandb.run.dir, "test_bin_counts.npy"), unique_counts)


# Allow for saving plots on remote machine
matplotlib.use('Agg')

MODELS = {
    "grud_joint": GRUModel, # Ignore graph structure, evolve single joint latent state
    "grud_node": GRUNodeModel, # Treat each node independently, independent latent state
    "tgnn4i": GRUGraphModel, # Utilizes graph structure
    "transformer_node": TransformerForecaster,
    "transformer_joint": TransformerJointForecaster,
}
if __name__ == "__main__":

    main()

