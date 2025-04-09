import streamlit as st
import sys

sys.path.append("C:/Users/Admin/Desktop/Omdena/Nepal_CBW_Batch2/NIC2_A/Week 12")
from visualizations import plot_time_series, plot_seasonal_patterns, plot_yearly_trends

def show(df):
    """
    Display the data exploration page
    """
    st.header("Data Exploration")

    # Show the raw data
    st.subheader("Raw Temperature Data")
    st.dataframe(df.head(10))

    # basic statistics
    st.subheader("Statistical Summary")
    st.write(df['temperatures'].describe())

    # Time series plot
    st.subheader("Temperatures over time")
    fig = plot_time_series(df)
    st.pyplot(fig)

    # Seasonal Plot
    st.subheader("Seasonal Temperature pattern")
    fig = plot_seasonal_patterns(df)
    st.pyplot(fig)

    # Yearly average temperatures
    st.subheader("Yearly Average Temperatures")
    fig = plot_yearly_trends(df)
    st.pyplot(fig)