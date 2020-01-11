"""
Module to train and test different models for the Prophet predictor.
"""

import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt, exp

import data_preprocessor

BASE_FOLDER = 'PROPHET_MODELS/'
LOG_FOLDER = 'LOG_DATA/NEW/'
RAW_FOLDER = 'RAW_DATA/'


def fit_cycle(train_dataset, test_dataset, period, fourier_order, changepoint_scale=0.05, changepoints=25,
              log_data=False):
    print('FITTING ========================================')
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,
                    changepoint_prior_scale=changepoint_scale, n_changepoints=changepoints)
    model.add_seasonality(name='test_seasonality', period=period, fourier_order=fourier_order, mode='additive')
    model.fit(train_dataset)

    # The period is the number of days to predict into the future, counting weekend days and holidays
    future = model.make_future_dataframe(periods=4230)

    # We have to remove Saturdays and Sunday from the future dataset that will be generated.
    # This is done by droping the rows that do not exist in the original dataset (faster than via Prophet API)
    future = future.set_index('ds').join(full_dataset.set_index('ds').y).reset_index()
    future.dropna(inplace=True)
    future = future.drop('y', axis=1)

    print('PREDICTING =====================================')
    forecast = model.predict(future)

    # Error calculation
    metrics_df = forecast.set_index('ds')[['yhat']].join(test_dataset.set_index('ds').y).reset_index()
    metrics_df.dropna(inplace=True)
    rmse = sqrt(mean_squared_error(metrics_df.y, metrics_df.yhat))
    print('rmse = ' + str(rmse))

    print('PLOT  ==========================================')
    ax = plt.gca()
    train_dataset.plot(kind='line', x='ds', y='y', ax=ax, color='blue', label='train data', figsize=(16, 9))
    forecast.plot(kind='line', x='ds', y='yhat', ax=ax, color='red', label='model prediction')
    metrics_df.plot(kind='line', x='ds', y='y', ax=ax, color='green', label='test data')

    if log_data:
        # Return to original data
        metrics_df['y'] = metrics_df['y'].apply(lambda x: exp(x))
        metrics_df['yhat'] = metrics_df['yhat'].apply(lambda x: exp(x))
        train_dataset['y'] = train_dataset['y'].apply(lambda x: exp(x))
        forecast['yhat'] = forecast['yhat'].apply(lambda x: exp(x))

        forecast.to_csv(BASE_FOLDER + LOG_FOLDER + 'prophet_predictions_' + str(period) + '_' + str(fourier_order) + '_' +
                        str(changepoint_scale) + '_' + str(changepoints) + '.csv')

        # Error calculation back to original data
        rmse_exp = sqrt(mean_squared_error(metrics_df.y, metrics_df.yhat))
        print('rmse_exp = ' + str(rmse_exp))
        plt.ylabel('log(USD)')
        plt.xlabel('days')
        plt.title('period=' + str(period) + '|fourier order=' + str(fourier_order) + '|prior_scale=' + str(
            changepoint_scale) + '|n_changepoints=' + str(changepoints) + '|RMSE(log data)=' + str(rmse)  +
                  '|RMSE(original  data)=' + str(rmse_exp))
        plt.savefig(BASE_FOLDER + LOG_FOLDER + 'PROPHET_MODEL_LOG_' + str(period) + '_' + str(fourier_order) + '_' +
                    str(changepoint_scale) + '_' + str(changepoints) + '.png')

    else:
        forecast.to_csv(BASE_FOLDER + RAW_FOLDER + 'prophet_predictions_' + str(period) + '_' + str(fourier_order)
                        + '_' + str(changepoint_scale) + '_' + str(changepoints) + '.csv')
        plt.ylabel('USD')
        plt.xlabel('days')
        plt.title('period=' + str(period) + '|fourier order=' + str(fourier_order) + '|prior_scale=' + str(
            changepoint_scale) + '|n_changepoints=' + str(changepoints) + '|RMSE=' + str(rmse))
        plt.savefig(BASE_FOLDER + RAW_FOLDER + 'PROPHET_MODEL_' + str(period) + '_' + str(fourier_order) + '_' +
                    str(changepoint_scale) + '_' + str(changepoints) + '.png')
    plt.clf()



if __name__ == "__main__":

    # Change this variable to perform or not a logarithmic conversion of the original data
    log_data = True

    full_dataset, train_ds, test_ds = data_preprocessor.prepare_data_prophet(log_data)

    # Grid definition. Add any desired values to the arrays
    grid = {
        "period": [5],
        "fourier_order": [5],
        "changepoint_scale": [0.05],
        "changepoints": [25]
    }

    for params in ParameterGrid(grid):
        # The original dataframes are copied to not be affected by the inner modifications in the method
        fit_cycle(train_ds.copy(), test_ds.copy(), params['period'], params['fourier_order'],
                  params['changepoint_scale'], params['changepoints'], log_data)

