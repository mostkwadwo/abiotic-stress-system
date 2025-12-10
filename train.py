import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AbioticStressDataset
from model import VisualStressModel

# CONSTANTS
CSV_PATH = "data/training_data_fused.csv"
BATCH_SIZE = 4 # Small batch for demo
EPOCHS = 5
LR = 0.001

def train():
    # 1. Setup Data
    print("Loading Dataset...")
    dataset = AbioticStressDataset(CSV_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Setup Model
    # 4 weather features: temp, humidity, solar, vpd
    model = VisualStressModel(num_weather_features=4)
    
    # Check for GPU (CUDA or MPS for Mac)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps") # Mac Metal Acceleration
    print(f"Training on: {device}")
    
    model.to(device)
    
    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() # Mean Squared Error for regression
    
    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            # Move data to GPU
            weather = batch['weather'].to(device)
            image = batch['image'].to(device)
            target = batch['target'].to(device)
            
            # Forward Pass
            optimizer.zero_grad()
            predictions = model(weather, image)
            
            # Loss Calculation
            loss = criterion(predictions, target)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

    # 5. Save Model
    torch.save(model.state_dict(), "data/stress_model.pth")
    print("Model saved to data/stress_model.pth")

if __name__ == "__main__":
    train()