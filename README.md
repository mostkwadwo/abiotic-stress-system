# Abiotic Stress Recognition System 🌾

**A Physics-Informed Spatiotemporal Deep Learning Model for Agritech.**

This system fuses satellite imagery (Spatial), weather history (Temporal), and thermodynamic equations (Physics) to predict plant stress (Drought/VPD) 7 days in advance.

## 🚀 Key Features

*   **Multi-Modal Fusion:** Combines **Sentinel-2** satellite data simulation with **OpenMeteo** weather streams.
*   **Physics-Informed Features:** Uses **JAX** to differentiably calculate Vapor Pressure Deficit (VPD) and Heat Stress Indices based on thermodynamic laws.
*   **Hybrid Architecture:** A **Visual-LSTM** (CNN + LSTM) that processes spatial crop health embeddings alongside temporal weather sequences.
*   **Interactive Dashboard:** A **Streamlit** app for real-time stress monitoring and forecasting.

## 🛠️ Tech Stack

*   **Core AI:** PyTorch (LSTM, CNN), JAX (Physics Engine).
*   **Data Engineering:** Azure Planetary Computer (STAC API), Pandas.
*   **Visualization:** Streamlit, Matplotlib.
*   **Environment:** Python 3.9+, CUDA/MPS Acceleration.

## 📊 Methodology

1.  **Ingestion:** Fetches daily weather data and aligns it with sparse satellite imagery passes.
2.  **Physics Engine:** Calculates `VPD = es - ea` (Tetens Equation) using JAX for high-performance numerical computing.
3.  **Model:** 
    *   *CNN Encoder* extracts features from the latest satellite image.
    *   *LSTM Decoder* fuses the image embedding with 14 days of weather history.
    *   *Head* predicts the next 7 days of stress values.

## 💻 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Pipeline:**
    ```bash
    python main.py
    ```

3.  **Train Model:**
    ```bash
    python train.py
    ```

4.  **Launch Dashboard:**
    ```bash
    streamlit run dashboard.py
    ```

## 📈 Results

The model outputs a 7-day forecast of the **Vapor Pressure Deficit (VPD)** index. 
*   **Red Line:** AI Prediction.
*   **Green Line:** Ground Truth.
*   **Grey Line:** 14-day Historical Context.