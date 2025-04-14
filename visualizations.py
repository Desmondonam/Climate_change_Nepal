import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_time_series(df):
    """
    Plot the temperatures over time
    """
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.plot(df['date'], df['temperatures'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperatures in C")
    ax.set_title("Monthly Temperatures ")
    ax.grid(True)
    return fig

def plot_seasonal_patterns(df):
    """
    plot for monthly temperature distributions
    """
    fig, ax = plt.subplots(figsize = (10, 6))
    sns.boxplot(x = 'month', y = 'temperatures', data = df, ax = ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperatures in C")
    ax.set_title("Monthly Temperatures Distributions")
    return fig


def plot_yearly_trends(df):
    """
    Plot the yearly average temperatures
    """
    fig, ax = plt.subplots(figsize = (10, 6))
    yearly_avg = df.groupby('year')['temperatures'].mean().reset_index()
    ax.plot(yearly_avg['year'], yearly_avg['temperatures'], marker = 'o')
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperatures in C")
    ax.set_title("Yearly Average Temperatures ")
    ax.grid(True)
    return fig

def plot_actual_vs_predicted(y_test, y_pred):
    """
    Plot the actual vs predicted values
    """
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.scatter(y_test, y_pred, alpha = 0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    ax.set_xlabel("Actual Temperatures")
    ax.set_ylabel("Predicted Temperatures in C")
    ax.set_title("Actual vs Predicted Temperatures ")
    return fig

def plot_prediction_context(hist_temps, pred_year, pred_month, prediction):
    """
    Plot for me the prediction in historical context
    """

    year_hist, temp_hist = zip(*hist_temps)

    fig, ax = plt.subplots(figsize = (10, 6))

    # Plot the historical data for the same month
    ax.scatter(year_hist, temp_hist, label = f"Historical (Month {pred_month})", color = "blue")
    ax.plot(year_hist, temp_hist, 'b--', alpha = 0.6)

    # Plot the prediction
    ax.scatter([pred_year], [prediction], color = 'red', s =100, label = "Prediction")

    # Add a trend line
    z = np.polyfit(year_hist, temp_hist, 1)
    p = np.poly1d(z)
    ax.plot(range(2010, pred_year + 1), p(range(2010, pred_year + 1)), 'g-', label = "Trend")

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Temperature for month {pred_month}")
    ax.set_title(f"Historical and predicted tempertures fo rhe moth of {pred_month} ")
    plt.legend()
    ax.grid(True)

    return fig