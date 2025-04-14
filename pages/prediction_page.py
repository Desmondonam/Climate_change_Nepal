import streamlit as st
import numpy as np
import sys

sys.path.append("C:/Users/Admin/Desktop/Omdena/Nepal_CBW_Batch2/NIC2_A/Week 12")
from models import load_model
from prediction import make_prediction, get_historical_context, get_historical_average
from visualizations import plot_prediction_context

def show(df):
    """
    Display the predictions page
    """
    st.header("Temperature Predictions")

    # Check if the model exists in the session
    if 'model' not in st.session_state:
        model = load_model()
        if model is None:
            st.warning("Mo trained model found. Please go to the model training page first")
            st.stop()

        st.session_state['model'] = model
        st.session_state['model_type'] = "Pre-trained Model"

    # Show which model is being used
    st.info(f"Using {st.session_state['model_type']} for predictions")

    # Prediction input
    st.subheader("Select data for prediction: ")
    pred_year = st.slider("Year", 2010, 2030, 2025)
    pred_month = st.slider("Month", 1, 12, 6)

    # make the prediction
    if st.button("Predict Temperature"):
        # Get model
        model = st.session_state['model']

        # make predictions
        prediction = make_prediction(model, pred_year, pred_month)

        # Display the results
        st.success(f"Predicted Temperature for {pred_year} - {pred_month:02d}: {prediction:.2f} °C")

        # Historical Comparison
        hist_avg = get_historical_average(df, pred_month)

        st.write(f"Historical Average for month {pred_month}: {hist_avg:.2f}°C")

        # calculate the difference from the historical average
        diff = prediction - hist_avg
        if diff > 0:
            st.write(f"Prediction is {diff:.2f}°C **higher** than historical average")
        else:
            st.write(f"Prediction is {abs(diff):.2f}°C **Lower** than historical average")

        # visualize prediction in context of historical data
        st.subheader("Prediction in Historical context")

        # Get the historical temperatures for the same month
        hist_temps = get_historical_context(df, pred_month)

        # Plot prediction context
        fig = plot_prediction_context(hist_temps, pred_year, pred_month, prediction)
        st.pyplot(fig)