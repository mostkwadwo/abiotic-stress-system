import torch
import torch.nn as nn

class VisualStressModel(nn.Module):
    def __init__(self, num_weather_features, hidden_dim=64, prediction_horizon=7):
        super(VisualStressModel, self).__init__()
        
        # --- ARM 1: VISION ENCODER (Spatial) ---
        # Processes the Satellite Image
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Smashes image to (Batch, 32, 1, 1)
            nn.Flatten() # Becomes (Batch, 32)
        )
        
        # --- ARM 2: TIME-SERIES DECODER (Temporal) ---
        # Fuses Weather Data + Image Embedding
        
        # The LSTM input size = Weather Features + Image Embedding Size (32)
        self.lstm_input_size = num_weather_features + 32
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # --- HEAD: PREDICTOR ---
        self.head = nn.Linear(hidden_dim, prediction_horizon)

    def forward(self, weather_seq, image_tensor):
        """
        weather_seq: (Batch, Time, Features)
        image_tensor: (Batch, Channels, H, W)
        """
        batch_size, seq_len, _ = weather_seq.shape
        
        # 1. Encode the Image
        # We assume the image represents the "static" state of the crop for this window
        img_embedding = self.cnn(image_tensor) # Output: (Batch, 32)
        
        # 2. Expand Image Embedding to match Time Sequence
        # We repeat the image info for every day in the weather sequence
        # (Batch, 32) -> (Batch, Time, 32)
        img_embedding_expanded = img_embedding.unsqueeze(1).repeat(1, seq_len, 1)
        
        # 3. Fuse Modalities
        # Concatenate Weather (Features) and Image (32) -> Combined Input
        fused_input = torch.cat((weather_seq, img_embedding_expanded), dim=2)
        
        # 4. Process with LSTM
        lstm_out, _ = self.lstm(fused_input)
        
        # 5. Predict using the final hidden state
        # We take the output of the last time step
        last_time_step = lstm_out[:, -1, :]
        prediction = self.head(last_time_step)
        
        return prediction