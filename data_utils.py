import torch
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import h5py
import pickle
import os
import sys

def z_score_scale(tensor, mean, std):
    """Applies z-score scaling using pre-calculated stats."""
    epsilon = 1e-8
    return (tensor - mean.to(tensor.device)) / (std.to(tensor.device) + epsilon)

def min_max_scale(tensor, min_vals, max_vals):
    """Applies min-max scaling using pre-calculated stats."""
    epsilon = 1e-8
    range_vals = max_vals - min_vals
    scaled_tensor = tensor.detach().clone()
    # Clamp values to the training range to handle outliers in val/test
    scaled_tensor.clamp_(min=min_vals.to(tensor.device), max=max_vals.to(tensor.device))
    return (scaled_tensor - min_vals.to(tensor.device)) / (range_vals.to(tensor.device) + epsilon)

def calculate_stats(hdf5_path, valid_keys, dataset_name='data', batch_size=1000):
    print(f"Starting stats calculation for: {hdf5_path}")
    total_count = 0
    sum_data = None
    sum_sq_data = None
    num_channels = None
    num_samples = len(valid_keys)

    with h5py.File(hdf5_path, 'r') as hf:
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_keys = valid_keys[i:end_idx]
            batch_data_list = [torch.from_numpy(hf[key][dataset_name][:]).float() for key in batch_keys]
            
            if not batch_data_list: continue
            batch_data = torch.stack(batch_data_list, dim=0)

            if batch_data.dim() < 5:
                batch_data = batch_data.unsqueeze(-1)
            
            batch_data[torch.isnan(batch_data)] = 0

            if num_channels is None:
                num_channels = batch_data.shape[-1]
                sum_data = torch.zeros(num_channels)
                sum_sq_data = torch.zeros(num_channels)

            batch_data_reshaped = batch_data.reshape(-1, num_channels)
            sum_data += torch.sum(batch_data_reshaped, dim=0)
            sum_sq_data += torch.sum(batch_data_reshaped ** 2, dim=0)
            total_count += batch_data_reshaped.shape[0]
            print(f"Processed {end_idx} of {num_samples} samples...")

    final_mean = (sum_data / total_count).view(1, 1, 1, 1, -1)
    variance = (sum_sq_data / total_count) - (final_mean ** 2)
    final_std = torch.sqrt(torch.clamp(variance, min=0)).view(1, 1, 1, 1, -1)
    return final_mean, final_std

def calculate_rolling_min_max(hdf5_path, valid_keys, dataset_name='data', batch_size=1000):
    print(f"Starting rolling min/max calculation for: {hdf5_path}")
    min_vals_channels, max_vals_channels = None, None
    num_samples = len(valid_keys)
    
    with h5py.File(hdf5_path, 'r') as hf:
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_keys = valid_keys[i:end_idx]
            batch_data = torch.stack([torch.from_numpy(hf[key][dataset_name][:]).float() for key in batch_keys])
            
            if batch_data.dim() < 5: batch_data = batch_data.unsqueeze(-1)
            batch_data[torch.isnan(batch_data)] = 0
            
            batch_min = torch.min(batch_data.reshape(batch_data.shape[0], -1, batch_data.shape[-1]), dim=1, keepdim=True)[0].squeeze()
            batch_max = torch.max(batch_data.reshape(batch_data.shape[0], -1, batch_data.shape[-1]), dim=1, keepdim=True)[0].squeeze()
            
            if min_vals_channels is None:
                min_vals_channels, max_vals_channels = batch_min, batch_max
            else:
                min_vals_channels = torch.min(min_vals_channels, batch_min)
                max_vals_channels = torch.max(max_vals_channels, batch_max)
            print(f"Processed {end_idx} of {num_samples} samples...")

    return min_vals_channels.view(1, 1, 1, 1, -1), max_vals_channels.view(1, 1, 1, 1, -1)

def calculate_filter_thresholds(hdf5_path, channel_idx=1, max_quantile=0.99, mean_quantile=0.99, magnitude_threshold=1):
    print("Calculating filtering thresholds...")
    per_sample_max_list, per_sample_mean_list = [], []
    with h5py.File(hdf5_path, 'r') as hf:
        keys = list(hf.keys())
        for key in keys:
            conc = hf[key]['data'][:, :, :, channel_idx]
            per_sample_max_list.append(np.amax(conc))
            per_sample_mean_list.append(np.mean(conc))
            
    max_threshold = np.quantile(np.array(per_sample_max_list), max_quantile) * magnitude_threshold
    return per_sample_max_list, max_threshold

class HDF5CombinedDataset(Dataset):
    def __init__(self, input_hdf5_path, output_hdf5_path, valid_keys, input_transform=None, output_transform=None):
        self.input_hdf5_path = input_hdf5_path
        self.output_hdf5_path = output_hdf5_path
        self.keys = sorted(valid_keys)
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.input_hf = None
        self.output_hf = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.input_hf is None:
            self.input_hf = h5py.File(self.input_hdf5_path, 'r')
            self.output_hf = h5py.File(self.output_hdf5_path, 'r')
        
        key = self.keys[idx]
        input_data = torch.from_numpy(self.input_hf[key]['data'][:]).float()
        output_data = torch.from_numpy(self.output_hf[key]['data'][:]).float()

        if self.input_transform: input_data = self.input_transform(input_data)
        if self.output_transform: output_data = self.output_transform(output_data)
        return input_data, output_data

def prepare_dataloaders(input_hdf5, output_hdf5, batch_size=2, norm_method='z-score', stats_path='normalization_stats'):
    os.makedirs(stats_path, exist_ok=True)
    filtered_keys_file = os.path.join(stats_path, 'filtered_sample_keys.pkl')

    if not os.path.exists(filtered_keys_file):
        per_max, threshold = calculate_filter_thresholds(output_hdf5)
        with h5py.File(output_hdf5, 'r') as hf:
            all_keys = list(hf.keys())
            valid_keys = [all_keys[i] for i, val in enumerate(per_max) if val < threshold]
        with open(filtered_keys_file, 'wb') as f: pickle.dump(valid_keys, f)
    else:
        with open(filtered_keys_file, 'rb') as f: valid_keys = pickle.load(f)

    if norm_method == 'z-score':
        in_p = os.path.join(stats_path, 'input_stats_z_score.pkl')
        out_p = os.path.join(stats_path, 'output_stats_z_score.pkl')
        if not os.path.exists(in_p):
            in_m, in_s = calculate_stats(input_hdf5, valid_keys)
            out_m, out_s = calculate_stats(output_hdf5, valid_keys)
            pickle.dump({'mean': in_m, 'std': in_s}, open(in_p, 'wb'))
            pickle.dump({'mean': out_m, 'std': out_s}, open(out_p, 'wb'))
        
        in_s_data = pickle.load(open(in_p, 'rb'))
        out_s_data = pickle.load(open(out_p, 'rb'))
        in_trans = lambda x: z_score_scale(x, in_s_data['mean'], in_s_data['std'])
        out_trans = lambda x: z_score_scale(x, out_s_data['mean'], out_s_data['std'])

    elif norm_method == 'min-max':
        in_p = os.path.join(stats_path, 'input_stats_min_max.pkl')
        out_p = os.path.join(stats_path, 'output_stats_min_max.pkl')
        if not os.path.exists(in_p):
            in_min, in_max = calculate_rolling_min_max(input_hdf5, valid_keys)
            out_min, out_max = calculate_rolling_min_max(output_hdf5, valid_keys)
            pickle.dump({'min': in_min, 'max': in_max}, open(in_p, 'wb'))
            pickle.dump({'min': out_min, 'max': out_max}, open(out_p, 'wb'))
        
        in_s_data = pickle.load(open(in_p, 'rb'))
        out_s_data = pickle.load(open(out_p, 'rb'))
        in_trans = lambda x: min_max_scale(x, in_s_data['min'], in_s_data['max'])
        out_trans = lambda x: min_max_scale(x, out_s_data['min'], out_s_data['max'])

    # 3. Create Splits and Loaders
    dataset = HDF5CombinedDataset(input_hdf5, output_hdf5, valid_keys, in_trans, out_trans)
    train_sz = int(0.8 * len(dataset))
    val_sz = int(0.1 * len(dataset))
    test_sz = len(dataset) - train_sz - val_sz
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_sz, val_sz, test_sz], generator=torch.Generator().manual_seed(0))
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )

import os
import torch
from data_utils import prepare_dataloaders

# --- User Settings ---
BATCH_SIZE = 2
NORMALIZATION_METHOD = 'z-score'  # 'z-score' or 'min-max'
INPUT_HDF5 = 'small_input_arrays.hdf5'
OUTPUT_HDF5 = 'small_output_arrays.hdf5'
STATS_PATH = 'normalization_stats'

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

training_loader, validation_loader, test_loader = prepare_dataloaders(
    input_hdf5=INPUT_HDF5,
    output_hdf5=OUTPUT_HDF5,
    batch_size=BATCH_SIZE,
    norm_method=NORMALIZATION_METHOD,
    stats_path=STATS_PATH
)

print(f"DataLoaders are ready. Train size: {len(training_loader.dataset)}")