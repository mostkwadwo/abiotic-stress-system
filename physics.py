import jax
import jax.numpy as jnp

# JAX allows these calculations to be JIT compiled and run on GPU
# This is "Differentiable Physics"

@jax.jit
def calculate_saturation_vapor_pressure(temp_c):
    """
    Computes Saturation Vapor Pressure (es) using the Tetens equation.
    temp_c: Temperature in Celsius
    """
    return 0.6108 * jnp.exp((17.27 * temp_c) / (temp_c + 237.3))

@jax.jit
def calculate_vpd(temp_c, relative_humidity):
    """
    Calculates Vapor Pressure Deficit (VPD) in kPa.
    
    VPD is the difference between the amount of moisture the air CAN hold
    vs what it ACTUALLY holds. High VPD = High Atmospheric Thirst (Drought Stress).
    """
    es = calculate_saturation_vapor_pressure(temp_c)
    ea = es * (relative_humidity / 100.0)
    return es - ea

@jax.jit
def calculate_heat_stress_index(temp_c, solar_rad):
    """
    A simplified heat stress proxy fusing temperature and solar radiation.
    (Custom logic can be added here for specific crop thresholds).
    """
    # Example: If temp > 30C and Solar Rad > 800 W/m2, stress increases non-linearly
    base_stress = jnp.maximum(0.0, temp_c - 30.0)
    rad_factor = jnp.maximum(0.0, solar_rad - 800.0) / 100.0
    return base_stress + (base_stress * rad_factor)

if __name__ == "__main__":
    # Test the physics engine
    temps = jnp.array([25.0, 32.0, 35.0])
    rh = jnp.array([60.0, 40.0, 30.0]) # Humidity dropping as it gets hotter
    
    vpds = calculate_vpd(temps, rh)
    print(f"Computed VPDs (kPa): {vpds}")
    # Output should show increasing values (Higher is worse for plants)