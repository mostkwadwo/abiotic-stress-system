import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import VisualStressModel
from dataset import AbioticStressDataset

# Page Config
st.set_page_config(page_title="Abiotic Stress AI", layout="wide")

# Constants
MODEL_PATH = "data/stress_model.pth"
CSV_PATH = "data/training_data_fused.csv"

# --- 1. Load Resources (Cached for Speed) ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    model = VisualStressModel(num_weather_features=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

@st.cache_data
def load_data():
    # Load dataset wrapper
    ds = AbioticStressDataset(CSV_PATH)
    return ds

# --- 2. UI Layout ---
st.title("🌱 Abiotic Stress Recognition System")
st.markdown("""
**Physics-Informed AI for Early Drought Detection.**
Fusing *Sentinel-2 Satellite Imagery* + *Weather Data* to predict crop stress 7 days ahead.
""")

col1, col2 = st.columns([1, 2])

# Load Logic
try:
    model, device = load_model()
    dataset = load_data()
    
    with col1:
        st.subheader("Simulation Controls")
        # Slider to pick a specific day in the dataset
        sample_idx = st.slider("Select Timeline Day", 
                               min_value=0, 
                               max_value=len(dataset)-1, 
                               value=10)
        
        # Get Data
        sample = dataset[sample_idx]
        
        # Display Current Conditions
        # (Un-scaling the normalized data for display is complex, showing raw values for now)
        st.info(f"📅 Prediction Window: Day {sample_idx} to {sample_idx+7}")
        st.write("Current Input Features (Normalized):")
        st.json({
            "Avg Temp": f"{sample['weather'][:, 0].mean():.2f}",
            "Avg Humidity": f"{sample['weather'][:, 1].mean():.2f}",
            "Current Stress (VPD)": f"{sample['weather'][-1, 3]:.2f}"
        })

    with col2:
        st.subheader("Stress Forecast (Next 7 Days)")
        
        # Run Inference
        weather_input = sample['weather'].unsqueeze(0).to(device)
        image_input = sample['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(weather_input, image_input)
            
        pred_numpy = prediction.cpu().numpy()[0]
        actual_numpy = sample['target'].numpy()
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot Context (Past 14 Days)
        days_past = range(-14, 0)
        past_vpd = sample['weather'][:, 3].numpy()
        ax.plot(days_past, past_vpd, label='History (Observed)', color='gray', linestyle='--')
        
        # Plot Future
        days_future = range(0, 7)
        ax.plot(days_future, actual_numpy, label='Actual Ground Truth', color='green', marker='o', alpha=0.5)
        ax.plot(days_future, pred_numpy, label='AI Prediction', color='red', marker='x', linewidth=2)
        
        # Formatting
        ax.axvline(x=0, color='black', linestyle=':', label='Today')
        ax.set_ylabel("VPD Stress Index (Normalized)")
        ax.set_xlabel("Days relative to Today")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Alert Logic
        avg_pred_stress = pred_numpy.mean()
        if avg_pred_stress > 0.6: # Arbitrary threshold for demo
            st.error(f"⚠️ HIGH STRESS ALERT: Predicted Index {avg_pred_stress:.2f}")
        else:
            st.success(f"✅ STABLE: Predicted Index {avg_pred_stress:.2f}")

except Exception as e:
    st.error(f"Error loading model or data. Did you run train.py? Error: {e}")