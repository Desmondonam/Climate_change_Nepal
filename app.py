import streamlit as st
from data_utils import load_data
import sys

sys.path.append("C:/Users/Admin/Desktop/Omdena/Nepal_CBW_Batch2/NIC2_A/Week 12/pages")
from pages import data_exploration, model_training, prediction_page

# Set the page configuration
st.set_page_config(
    page_title = "Climate Trend Predictor",
    page_icon=' ',
    layout = 'wide'
)

# Give the title
st.title("Climate Trend Analysis and Predictions")
st.markdown("Analyze historical Temperatures data and predict future trends")

df = load_data()


# Give the sidebar for the app navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", 'Prediction'])


# Display the selected page
if page == "Data Exploration":
    data_exploration.show(df)
elif page == "Model Training":
    model_training.show(df)
else: # Prediction page
    prediction_page.show(df)