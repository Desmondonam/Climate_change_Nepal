import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle # Saving my model

def split_data(X, y, test_size = 0.2):
    """
    Split the data intoooo training and testing sets
    """
    return train_test_split(X, y, test_size=test_size, random_state= 42)

def train_model(X_train, y_train, model_type = "Linear Regression"):
    """
    train the model based on the specified type
    """
    if model_type == "Linear Regression":
        model = LinearRegression()

    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model and return the metrics
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate the metrics
    metrics = {
        'train_rmse' : np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse' : np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2' : r2_score(y_train, y_pred_train),
        'test_r2' : r2_score(y_test, y_pred_test),
        'y_test' : y_test,
        'y_pred_test': y_pred_test
    }

    return metrics

def save_model(model, file_name = 'climate_model.pkl'):
    """
    Save model to disk
    """
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_name = 'climate_model.pkl'):
    """
    Load the model that was saved
    """
    try:
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None

