import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AbioticStressDataset(Dataset):
    def __init__(self, csv_file, sequence_length=14, prediction_horizon=7):
        """
        Args:
            csv_file: Path to the fused data CSV.
            sequence_length: How many past days the model sees (Time steps).
            prediction_horizon: How many future days we want to predict.
        """
        self.data = pd.read_csv(csv_file)
        self.seq_len = sequence_length
        self.pred_len = prediction_horizon
        
        # Normalize the weather features (Critical for Neural Networks)
        feature_cols = ['temp_max', 'humidity_mean', 'solar_rad', 'vpd_kpa']
        self.features = self.data[feature_cols].values.astype(np.float32)
        
        # Simple MinMax Scaling (in production, use scikit-learn scaler and save it)
        self.min_vals = self.features.min(axis=0)
        self.max_vals = self.features.max(axis=0)
        self.features = (self.features - self.min_vals) / (self.max_vals - self.min_vals + 1e-6)
        
        # Target: We want to predict VPD (Drought Stress)
        self.target = self.data['vpd_kpa'].values.astype(np.float32)

    def __len__(self):
        # We stop when we run out of data for the prediction horizon
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        # 1. Get Weather Sequence (Past 14 days)
        # Shape: (Sequence_Length, Num_Features)
        weather_x = self.features[idx : idx + self.seq_len]
        
        # 2. Get "Satellite Image"
        # REAL WORLD: You would use rasterio to load the .tif file corresponding to dates[idx]
        # PORTFOLIO SIMULATION: We generate a random tensor to prove the Architecture works 
        # without downloading 50GB of images right now.
        # Shape: (Channels, Height, Width) -> Sentinel-2 has ~12 bands, we use 3 for demo
        image_x = torch.randn(3, 64, 64) 
        
        # 3. Get Target (Future 7 days of VPD)
        target_y = self.target[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        
        return {
            'weather': torch.tensor(weather_x),
            'image': image_x,
            'target': torch.tensor(target_y)
        }