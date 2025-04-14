import pandas as pd
import numpy as np

def make_prediction(model, year, month):
    """
    Make the temperature prediction for a given month or year
    """
    features = np.array([[year, month]])
    return model.predict(features)[0]

def get_historical_context(df, month):
    """
    Get historical temperatures for the same month
    """
    years = df['year'].unique()
    hist_temps = []

    for year in years:
        month_data = df[(df['year'] == year) & (df['month'] == month)]
        if not month_data.empty:
            hist_temps.append((year, month_data['temperatures'].values[0]))
    
    return hist_temps

def get_historical_average(df, month):
    """
    Get historical average temperatures of a given month
    """

    return df[df['month'] == month]['temperatures'].mean()