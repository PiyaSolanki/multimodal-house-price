from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return rmse, r2

from sklearn.ensemble import RandomForestRegressor

def train_rf_model(X, y):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model
