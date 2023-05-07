import pandas as pd
from fbprophet import Prophet
from Components.Data.data_handler import DataHandler
import itertools
import numpy as np
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric


class ProphetModel:
    def __init__(self, tickers, start, end, model=None):
        super().__init__()
        self.best_params = {}
        self.models_add = {}
        self.models_mult = {}
        self.data = DataHandler(tickers, start, end)
        self.param_grid = {
            'changepoint_prior_scale': [0.5],
            'seasonality_prior_scale': [10.0],
            }
        # self.param_grid = {
        #     'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5, 1.0],
        #     'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0, 20.0],
        #     'changepoint_range': [0.8, 0.85, 0.9, 0.95],
        #     'n_changepoints': [20, 25, 30, 35, 40],
        #     'seasonality_mode': ['additive', 'multiplicative'],
        #     'weekly_seasonality': [True, False],
        #     'daily_seasonality': [True, False],
        #     'yearly_seasonality': [True, False]
        #     }

    def train(self, data, target_cols=None):
        if target_cols is None:
            target_cols = data.columns

        for col in target_cols:
            # Prepare the data for Prophet
            df = data[[col]].reset_index()
            df.columns = ['ds', 'y']

            # Instantiate and fit the additive Prophet model
            model_add = Prophet()
            model_add.fit(df)
            self.models_add[col] = model_add

            # Instantiate and fit the multiplicative Prophet model
            model_mult = Prophet(seasonality_mode='multiplicative')
            model_mult.fit(df)
            self.models_mult[col] = model_mult


    def predict(self, future_periods, target_cols=None):
        if target_cols is None:
            target_cols = self.models_add.keys()

        predictions_add = {}
        predictions_mult = {}

        for col in target_cols:
            # Make predictions for the specified number of future periods
            future = self.models_add[col].make_future_dataframe(periods=future_periods)

            # Predictions using the additive model
            forecast_add = self.models_add[col].predict(future)
            predictions_add[col] = forecast_add['yhat'].tail(future_periods).values

            # Predictions using the multiplicative model
            forecast_mult = self.models_mult[col].predict(future)
            predictions_mult[col] = forecast_mult['yhat'].tail(future_periods).values

            # Choose the model with the lowest error (if the best_params dictionary is created)
            if col in self.best_params and 'additive' in self.best_params[col] and 'multiplicative' in self.best_params[col]:
                if self.best_params[col]['additive']['seasonality_mode'] == 'additive':
                    predictions_add[col] = predictions_add[col]
                else:
                    predictions_add[col] = predictions_mult[col]

        return predictions_add, predictions_mult



    def optimize_hyperparameters(self, data, target_col, initial=None, period=None, horizon=None, metric='mape'):
        if initial is None:
            initial = pd.to_timedelta(data['ds'].iloc[-1] - data['ds'].iloc[0]) * 0.5

        if period is None:
            period = pd.to_timedelta('365 days')

        if horizon is None:
            horizon = pd.to_timedelta('365 days')

        # Calculate all combinations of hyperparameters in the parameter grid
        keys, values = zip(*self.param_grid.items())
        all_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Train and evaluate models with different hyperparameters
        performance = []
        for params in all_params:
            model = Prophet(**params)
            model.fit(data)

            df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
            df_p = performance_metrics(df_cv, rolling_window=1)
            performance.append(df_p[metric].values[0])

        # Find the best hyperparameters
        best_params = all_params[np.argmin(performance)]
        
        # Train the model with the best hyperparameters on the entire dataset
        best_model = Prophet(**best_params)
        best_model.fit(data)

        # Train the best additive model
        best_params_add = best_params.copy()
        best_params_add['seasonality_mode'] = 'additive'
        best_model_add = Prophet(**best_params_add)
        best_model_add.fit(data)

        # Train the best multiplicative model
        best_params_mult = best_params.copy()
        best_params_mult['seasonality_mode'] = 'multiplicative'
        best_model_mult = Prophet(**best_params_mult)
        best_model_mult.fit(data)

        # Add the best parameters for both additive and multiplicative models to the dictionary
        self.best_params[target_col] = {
            'additive': best_params_add,
            'multiplicative': best_params_mult
        }

        return self.best_params[target_col], best_model
    
    def train_best_models(self, data, target_cols=None, best_params=None):
        if target_cols is None:
            target_cols = data.columns

        for col in target_cols:
            # Prepare the data for Prophet
            df = data[[col]].reset_index()
            df.columns = ['ds', 'y']

            # Instantiate and fit the best additive Prophet model
            best_params_add = best_params.copy()
            best_params_add['seasonality_mode'] = 'additive'
            model_add = Prophet(**best_params_add)
            model_add.fit(df)
            self.models_add[col] = model_add

            # Instantiate and fit the best multiplicative Prophet model
            best_params_mult = best_params.copy()
            best_params_mult['seasonality_mode'] = 'multiplicative'
            model_mult = Prophet(**best_params_mult)
            model_mult.fit(df)
            self.models_mult[col] = model_mult

    def create_future_dataframe(self, stock_data, future_periods, target_cols=None):  # Add 'stock_data' parameter
        if target_cols is None:
            target_cols = self.models_add.keys()

        # Create a dataframe with index as future dates
        last_date = pd.to_datetime(self.data.stock_data.index[-1])  # Convert the last_date to a datetime object
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_periods)
        future_df = pd.DataFrame(index=future_dates, columns=target_cols)

        # Fill the dataframe with predictions from the best models
        predictions_add, predictions_mult = self.predict(future_periods, target_cols)

        for col in target_cols:
            # Choose the model with the lowest error (assuming the best_params dictionary is created)
            if self.best_params[col]['additive']['seasonality_mode'] == 'additive':
                print("Additive")
                future_df[col] = predictions_add[col]
            else:
                print("Multiplicative")
                future_df[col] = predictions_mult[col]

        return future_df
