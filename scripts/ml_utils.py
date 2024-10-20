import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

# def get_metrics_pd(y_values, preds, model_name):
#     mse = np.round(mean_squared_error(y_values, preds),3)
#     r2 = np.round(r2_score(y_values, preds),3)
#     mae = np.round(mean_absolute_error(y_values, preds),3)
#     rmse = np.round(np.sqrt(mse),3)
#     return pd.Series([mse, r2, mae, rmse], index=['MSE', 'R^2', 'MAE', 'RMSE'])

# def get_metrics_pd(y_values, preds, model_name):
#     mse = np.round(mean_squared_error(y_values, preds),2)
#     r2 = np.round(r2_score(y_values, preds),3)
#     mae = np.round(mean_absolute_error(y_values, preds),2)
#     rmse = np.round(np.sqrt(mse),2)
#     return pd.DataFrame([mse, r2, mae, rmse], index=['MSE', 'R^2', 'MAE', 'RMSE'], columns=[f'{model_name}'])

def get_metrics_pd(y_values, preds, model_name):
    mse = f"{np.round(mean_squared_error(y_values, preds), 2):,.2f}"
    r2 = f"{np.round(r2_score(y_values, preds), 3):,.3f}"
    mae = f"{np.round(mean_absolute_error(y_values, preds), 2):,.2f}"
    rmse = f"{np.round(np.sqrt(mean_squared_error(y_values, preds)), 2):,.2f}"
    return pd.DataFrame(
        [mse, r2, mae, rmse], 
        index=['MSE', 'R^2', 'MAE', 'RMSE'], 
        columns=[f'{model_name}']
    )
