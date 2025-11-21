import streamlit as st
import time

# Define the processing steps
steps = [
    "Fetching elevation data",
    "Processing elevation data",
    "Plotting street grades",
    "Fetching OSM features"
]

# Create placeholders for each step
step_placeholders = [st.empty() for _ in steps]

# Display initial steps with pending symbol
for placeholder, step in zip(step_placeholders, steps):
    placeholder.markdown(f"⏳ {step}")

# Function to simulate a step
def run_step(index, func, *args, **kwargs):
    with st.spinner(f"{steps[index]}..."):
        result = func(*args, **kwargs)
    step_placeholders[index].markdown(f"✅ {steps[index]}")
    return result

# Dummy functions to simulate processing
def fetch_elevation():
    time.sleep(2)
    return "elevation_data"

def process_elevation(data):
    time.sleep(2)
    return "processed_data"

def plot_grades(data):
    time.sleep(2)
    return "plot_ready"

def fetch_osm_features():
    time.sleep(2)
    return "osm_features"

# Run steps one by one
elevation = run_step(0, fetch_elevation)
processed = run_step(1, process_elevation, elevation)
plot = run_step(2, plot_grades, processed)
osm = run_step(3, fetch_osm_features)