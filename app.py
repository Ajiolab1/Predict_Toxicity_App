import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("rf_model.pkl")


# Define LC50 estimation function
def estimate_lc50(model, metal_type, exposure_time, temp, ph, do, step=500):
    import numpy as np
    import pandas as pd

    metal_encoded = 0 if metal_type == "Zn" else 1
    concentrations = np.linspace(0, 100, step)
    df = pd.DataFrame({
        'Metal_Type': [metal_encoded] * step,
        'Concentration_mg_L': concentrations,
        'Exposure_Time_h': [exposure_time] * step,
        'Temp_C': [temp] * step,
        'pH': [ph] * step,
        'DO_mg_L': [do] * step
    })

    predictions = model.predict(df.values)
    closest_idx = np.argmin(np.abs(predictions - 50))
    return concentrations[closest_idx]
# Function to generate LC50 chart data
def generate_lc50_curve_data(model, metal_type, exposure_time, temp, ph, do, step=500):
    import numpy as np
    concentrations = np.linspace(0, 100, step)
    metal_encoded = 0 if metal_type == "Zn" else 1
    df = pd.DataFrame({
        'Metal_Type': [metal_encoded] * step,
        'Concentration_mg_L': concentrations,
        'Exposure_Time_h': [exposure_time] * step,
        'Temp_C': [temp] * step,
        'pH': [ph] * step,
        'DO_mg_L': [do] * step
    })
    predictions = model.predict(df.values)
    lc50_estimate = concentrations[np.argmin(np.abs(predictions - 50))]
    return concentrations, predictions, lc50_estimate

# App title and intro
st.title("🐟 Predict Toxicity in African Catfish")
st.write("This app predicts mortality (%) in African catfish exposed to heavy metals.")

# Collect user input
metal = st.radio("Select Metal Type", ["Zn", "Cu"])
concentration = st.number_input("Concentration (mg/L)", min_value=0.0, step=0.1)
exposure_time = st.selectbox("Exposure Time (hours)", [24, 48, 72, 96])
temperature = st.slider("Temperature (°C)", 20.0, 35.0, 27.0)
ph = st.slider("pH", 5.0, 9.0, 7.0)
do = st.slider("Dissolved Oxygen (mg/L)", 2.0, 10.0, 5.5)

# Encode metal
metal_encoded = 0 if metal == "Zn" else 1

# Create input dataframe
input_data = pd.DataFrame([[metal_encoded, concentration, exposure_time, temperature, ph, do]],
                          columns=['Metal_Type', 'Concentration_mg_L', 'Exposure_Time_h', 'Temp_C', 'pH', 'DO_mg_L'])

# Predict button
if st.button("Predict Mortality and Estimate LC₅₀"):
    result = model.predict(input_data.values)[0]
    st.success(f"Predicted Mortality: {result:.2f}%")

    # Estimate LC50 based on current inputs
    lc50 = estimate_lc50(
        model=model,
        metal_type=metal,
        exposure_time=exposure_time,
        temp=temperature,
        ph=ph,
        do=do
    )
    st.info(f"Estimated LC₅₀: {lc50:.2f} mg/L")

# Plot LC50 curve
concs, preds, lc50_val = generate_lc50_curve_data(
    model=model,
    metal_type=metal,
    exposure_time=exposure_time,
    temp=temperature,
    ph=ph,
    do=do
)

fig, ax = plt.subplots()
ax.plot(concs, preds, label="Predicted Mortality (%)", color="blue")
ax.axhline(50, color='red', linestyle='--', label="50% Mortality Line")
ax.axvline(lc50_val, color='green', linestyle='--', label=f"LC₅₀ ≈ {lc50_val:.2f} mg/L")
ax.set_xlabel("Concentration (mg/L)")
ax.set_ylabel("Mortality (%)")
ax.set_title("Dose-Response Curve")
ax.legend()
ax.grid(True)

st.pyplot(fig)
