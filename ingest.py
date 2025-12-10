import requests
import pandas as pd
from pystac_client import Client
import planetary_computer
from datetime import datetime, timedelta

# CONSTANTS
# Example Location: A farm field in California (Central Valley)
LAT = 36.6
LON = -120.6
START_DATE = "2023-06-01"
END_DATE = "2023-08-01"

def fetch_weather_data(lat, lon, start_date, end_date):
    """
    Fetches daily weather aggregation from OpenMeteo.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "relative_humidity_2m_mean", "shortwave_radiation_sum"],
        "timezone": "auto"
    }
    
    print(f"Fetching Weather Data for {lat}, {lon}...")
    response = requests.get(url, params=params)
    data = response.json()
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": data['daily']['time'],
        "temp_max": data['daily']['temperature_2m_max'],
        "humidity_mean": data['daily']['relative_humidity_2m_mean'],
        "solar_rad": data['daily']['shortwave_radiation_sum']
    })
    df['date'] = pd.to_datetime(df['date'])
    return df

def fetch_satellite_metadata(lat, lon, start_date, end_date):
    """
    Queries Azure Planetary Computer to find available Sentinel-2 imagery.
    We don't download the heavy images yet, just check availability to align dates.
    """
    print("Querying Azure Planetary Computer for Sentinel-2...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Search for Sentinel-2 Level-2A (Atmospherically Corrected)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[lon-0.01, lat-0.01, lon+0.01, lat+0.01],
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 20}} # Only clear images
    )

    items = search.item_collection()
    print(f"Found {len(items)} satellite passes.")
    
    sat_dates = []
    for item in items:
        # Get date and URL to the visual asset (for future use)
        dt = pd.to_datetime(item.datetime).normalize().tz_localize(None)
        sat_dates.append({
            "date": dt,
            "satellite_id": item.id,
            "asset_link": item.assets["visual"].href
        })
        
    return pd.DataFrame(sat_dates)

def fuse_data(weather_df, sat_df):
    """
    The Fusion Step: Merges daily weather with sparse satellite dates.
    """
    # Merge weather with satellite data
    # Left join on weather because weather is continuous (daily), satellite is sparse (every 5-10 days)
    fused_df = pd.merge(weather_df, sat_df, on="date", how="left")
    
    # Feature Engineering: Add a flag for when we have image data
    fused_df['has_satellite_image'] = fused_df['satellite_id'].notnull().astype(int)
    
    return fused_df