"""
Simulates a trader in the stock market during the test dataset for different models:
1. LSTM
2. Prophet
3. Other implemented trading strategies
"""

from keras.models import load_model
import lstm
import data_preprocessor
import prophet
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt


class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    ENDC = '\033[0m'


class Trader:
    def __init__(self, initial_capital):
        self.capital = initial_capital
        self.stock = 0

    def buy_or_hold_order(self, current_price):
        """
        Simulates a buy in the market. The maximum number of shares are bought.
        """
        if self.capital >= current_price:
            # Both options are considered: stock was previously zero or different than zero:
            stock_to_buy = self.capital // current_price
            self.capital -= stock_to_buy * current_price
            self.stock += stock_to_buy
        #     print(Colors.GREEN+'REAL BUY ++++++++++++++++'+Colors.ENDC)
        # else:
        #     print(Colors.GREEN+'+++'+Colors.ENDC)

    def sell_order(self, current_price):
        """
        Simulates a sell order in case the trader holds any stock.
        """
        if self.stock > 0:
            self.capital += self.stock * current_price
            self.stock = 0
        #     print(Colors.BLUE+'REAL SELL --------------------------------'+Colors.ENDC)
        # else:
        #     print(Colors.BLUE+'---'+Colors.ENDC)


def calculate_roi(current_value, cost):
    return (current_value - cost) / cost


def simulate_trade(real_values, predictions, initial_capital):
    """
    Simulates any trading entity, given an array of real prices and predictions
    """
    trader = Trader(initial_capital)
    for day in range(len(predictions)):
        if predictions[day] > real_values[day]:
            trader.buy_or_hold_order(real_values[day])
        else:
            trader.sell_order(real_values[day])

    # At the end of the dataset, a sell order is placed to convert all stocks to liquid with the price of the last
    # observation:
    trader.sell_order(real_values[len(predictions) - 1])
    return calculate_roi(trader.capital, initial_capital)


def simulate_trade_buy_hold(real_prices, initial_capital):
    """
    Simulates a trading entity which follows a simple buy and hold strategy
    """
    trader = Trader(initial_capital)
    trader.buy_or_hold_order(real_prices[0])
    trader.sell_order(real_prices[len(real_prices) - 1])
    return calculate_roi(trader.capital, initial_capital)


def simulate_trade_random(real_prices, initial_capital):
    """
    Simulates trading entity following a random behaviour
    """
    trader = Trader(initial_capital)
    for day in range(len(real_prices)):
        if random.choice([True, False]):
            trader.buy_or_hold_order(real_prices[day])
        else:
            trader.sell_order(real_prices[day])
    return calculate_roi(trader.capital, initial_capital)


def predict_average_method(real_prices, days_window):
    """
    Makes the predictions for the SMA method
    """
    predictions = []
    for day in range(len(real_prices)):
        predictions.append(calculate_mean(day, days_window, real_prices))
    return np.array(predictions)


def calculate_mean(index, elems_before, array):
    min_index = max(0, index - elems_before)
    max_index = index
    sum = 0
    # max_index + 1 since the current day is also included in the calculation
    for elem in range(min_index, max_index + 1):
        sum += array[elem]
    return sum / len(range(min_index, max_index + 1))


if __name__ == "__main__":

    initial_capital = 10000
    input_seq_length = 5  # Timesteps: must be changed depending on the model.

    # LSTM and Prophet models to execute
    lstm_model = 'model_5_500_8_96.h5'
    prophet_model = 'prophet_predictions_9000_5_0.13_25.csv'

    # Since the Prophet prediction DataFrame contains all the predictions and not only the ones in the test dataset,
    # it will be needed to extract only the ones used for the rest of the model:
    initial_date_prophet = 11671

    # Prepares the test and train dataser for several models to use
    X_train, y_train, X_test, y_test = data_preprocessor.prepare_data_lstm(input_seq_length, lstm.SCALER,
                                                                           lstm.N_FEATURES)

    # LSTM MODEL ========================================================================
    model = load_model(lstm.MODELS_FOLDER + lstm_model)
    rmse, y_test_scaled, predictions_lstm = lstm.predict_model(model, X_test, y_test, lstm.SCALER)
    roi_lstm = simulate_trade(y_test_scaled, predictions_lstm, initial_capital)
    print('ROI(LSTM) = ' + str(roi_lstm))


    # PROPHET MODEL =======================================================================
    df_predictions_prophet = pd.read_csv(
        prophet.BASE_FOLDER + prophet.RAW_FOLDER + prophet_model)
    # We select only from 2008-05-14 to 2019-12-05, which are the dates tested with the LSTM
    predictions_prophet = df_predictions_prophet[initial_date_prophet:]
    predictions_prophet = np.array(predictions_prophet[['yhat']])
    roi_prophet = simulate_trade(y_test_scaled, predictions_prophet, initial_capital)
    print('ROI(Facebook Prophet) = ' + str(roi_prophet))


    # AVERAGE MODEL ======================================================================
    sliding_window = 20  # Days to include in the average (current day also included)
    predictions_average = predict_average_method(y_test_scaled, sliding_window)
    roi_average = simulate_trade(y_test_scaled, predictions_average, initial_capital)
    print('ROI(SMA ' + str(sliding_window) + ' DAYS) = ' + str(roi_average))


    # AVERAGE MODEL ======================================================================
    sliding_window = 60  # Days to include in the average (current day also included)
    predictions_average_2 = predict_average_method(y_test_scaled, sliding_window)
    roi_average_2 = simulate_trade(y_test_scaled, predictions_average_2, initial_capital)
    print('ROI(SMA ' + str(sliding_window) + ' days) = ' + str(roi_average_2))


    # BUY AND HOLD MODEL ==================================================================
    roi_bh = simulate_trade_buy_hold(y_test_scaled, initial_capital)
    print('ROI(Buy&Hold) = ' + str(roi_bh))


    # RANDOM MODEL (mean of several simulations) ===========================================
    simulations = 1000
    roi_random = 0
    for i in range(simulations):
        roi_random += simulate_trade_random(y_test_scaled, initial_capital)
    print('ROI(random) = ' + str(roi_random / simulations))


    # LAST VALUE MODEL =====================================================================
    # In this model, the prediction for the next day is the Nth-previous day closing price
    shift_observations = 5
    predictions_last_value = np.roll(y_test_scaled, shift_observations)
    roi_last_value = simulate_trade(y_test_scaled, predictions_last_value, initial_capital)
    print('ROI(last value) = ' + str(roi_last_value))

    # RMSE between the predictions and a shif of N observations
    # print('\tRMSE(predictions, predictions-'+str(shift_observations)+') = '+str(sqrt(mean_squared_error(y_test_scaled, predictions_last_value))))
