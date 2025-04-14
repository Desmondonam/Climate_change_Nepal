import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data():
    """
    Generate or load data (Climate).
    In real world you will go ahead and load your CSV data or read your API data
    """  
    # Create data for year of months - temperature averages
    dates = pd.date_range(start = '2010-01-01', end ='2023-12-31', freq='M')

    # Generate some synthetic data
    temps = []
    for i in range(len(dates)):
        # Base my temperatures with seasonal patterns
        seasonal = 15 + 10 * np.sin(2 * np.pi * i /12)
        # Add an upward trend
        trend = 0.03 * 1
        # Add some random noise to the data
        noise = np.random.normal(0, 1.5)
        temps.append(seasonal + trend + noise)

    # create the df
    df = pd.DataFrame({
        "date" : dates,
        "temperatures": temps
    })

    # Extract the features 

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    return df


def prepare_features(df):
    """
    Prepare features for model training

    """
    X = df[['year', 'month']].values
    y = df['temperatures'].values

    return X, y
