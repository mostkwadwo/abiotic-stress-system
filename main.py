import pandas as pd
import jax.numpy as jnp
from ingest import fetch_weather_data, fetch_satellite_metadata, fuse_data, LAT, LON, START_DATE, END_DATE
from physics import calculate_vpd, calculate_heat_stress_index

def main():
    # 1. Ingest Data
    weather_df = fetch_weather_data(LAT, LON, START_DATE, END_DATE)
    sat_df = fetch_satellite_metadata(LAT, LON, START_DATE, END_DATE)
    
    # 2. Fuse Data (Temporal Alignment)
    dataset = fuse_data(weather_df, sat_df)
    
    # 3. Apply JAX Physics Engine
    # Convert pandas series to JAX arrays
    temps = jnp.array(dataset['temp_max'].values)
    rh = jnp.array(dataset['humidity_mean'].values)
    solar = jnp.array(dataset['solar_rad'].values)
    
    print("Calculating Physics Indices with JAX...")
    
    # Calculate VPD (Drought Indicator)
    dataset['vpd_kpa'] = calculate_vpd(temps, rh)
    
    # Calculate Heat Stress (Heat Indicator)
    dataset['heat_stress_idx'] = calculate_heat_stress_index(temps, solar)
    
    # 4. Preview Data for the ML Model
    print("\n--- Final Dataset Snippet ---")
    print(dataset[['date', 'temp_max', 'vpd_kpa', 'heat_stress_idx', 'has_satellite_image']].head(10))
    
    # 5. Save for Phase 2
    dataset.to_csv("training_data_fused.csv", index=False)
    print("\nData pipeline complete. Saved to training_data_fused.csv")

if __name__ == "__main__":
    main()