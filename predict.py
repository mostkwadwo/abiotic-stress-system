import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import VisualStressModel
from dataset import AbioticStressDataset

# CONSTANTS
MODEL_PATH = "data/stress_model.pth"
CSV_PATH = "data/training_data_fused.csv"

def predict_and_plot():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # 2. Load the Saved Model
    print("Loading model...")
    # Re-initialize the architecture
    model = VisualStressModel(num_weather_features=4) 
    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode (turns off dropout)

    # 3. Load Data Sample
    # We use the dataset class to handle the normalization/windowing
    dataset = AbioticStressDataset(CSV_PATH)
    
    # Let's pick a random day from the dataset to test
    sample_idx = 10 
    sample = dataset[sample_idx]
    
    # Prepare inputs (Add Batch Dimension [1, ...])
    weather_input = sample['weather'].unsqueeze(0).to(device) # (1, 14, 4)
    image_input = sample['image'].unsqueeze(0).to(device)     # (1, 3, 64, 64)
    actual_future = sample['target'].numpy()                  # True values

    # 4. Make Prediction
    print("Running Inference...")
    with torch.no_grad():
        prediction_tensor = model(weather_input, image_input)
    
    predicted_future = prediction_tensor.cpu().numpy()[0]

    # 5. Visualize Results
    print("Generating Forecast Plot...")
    
    days_past = range(-14, 0)
    days_future = range(0, 7)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Past Context (using the normalized input data for visualization)
    # Feature 3 is VPD in our dataset (index 3)
    past_vpd = sample['weather'][:, 3].numpy()
    plt.plot(days_past, past_vpd, label='Past 14 Days (Observed)', color='gray', linestyle='--')
    
    # Plot Actual Future
    plt.plot(days_future, actual_future, label='Actual Next 7 Days', color='green', marker='o')
    
    # Plot Predicted Future
    plt.plot(days_future, predicted_future, label='AI Forecast', color='red', marker='x', linewidth=2)
    
    plt.title("Abiotic Stress Forecast (Drought/VPD)")
    plt.xlabel("Days (Relative to Today)")
    plt.ylabel("VPD (Normalized Stress Index)")
    plt.axvline(x=0, color='black', linestyle=':', label='Today')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig("data/stress_forecast.png")
    print("Success! Chart saved to 'data/stress_forecast.png'")

if __name__ == "__main__":
    predict_and_plot()