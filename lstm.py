"""
Module to train and test different models for the LSTM predictor.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from math import sqrt
import csv
import os
import data_preprocessor
from sklearn.model_selection import ParameterGrid


# Constants for the LSTM model
OUTPUT_SEQ_LENGTH = 1
DROPOUT_PROB = 0.2
BASE_MODEL_FILENAME = 'model'
MODELS_FOLDER = 'LSTM_FINAL_MODELS/'
IMG_FOLDER = MODELS_FOLDER + 'IMG/'
N_FEATURES = 1  # univariate (Close prices only)
SCALER = MinMaxScaler(feature_range=(0, 1))


def predict_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    predictions = predictions.flatten()
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    rmse = sqrt(mean_squared_error(y_test_scaled, predictions))
    return rmse, y_test_scaled, predictions


def calculate_rmse(input_seq_length, number_units, number_epochs, batch_size):
    """
    Builds or loads an LSTM model and returns its RMSE
    """

    model_filename = BASE_MODEL_FILENAME + '_' + str(input_seq_length) + '_' + str(number_units) + '_' + \
                     str(number_epochs) + '_' + str(batch_size) + '.h5'
    X_train, y_train, X_test, y_test = data_preprocessor.prepare_data_lstm(input_seq_length, SCALER, N_FEATURES)

    if not os.path.exists(MODELS_FOLDER + model_filename):
        model = Sequential()
        model.add(LSTM(units=number_units, return_sequences=True, input_shape=(X_train.shape[1], N_FEATURES)))
        model.add(Dropout(DROPOUT_PROB))
        model.add(LSTM(units=number_units))
        model.add(Dropout(DROPOUT_PROB))
        model.add(Dense(units=OUTPUT_SEQ_LENGTH))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

        model.fit(X_train, y_train, epochs=number_epochs, batch_size=batch_size)
        model.save(MODELS_FOLDER+model_filename)

    print('LOAD MODEL...')
    model = load_model(MODELS_FOLDER+model_filename)

    print('PREDICT MODEL...')
    rmse, y_test_scaled, predictions = predict_model(model, X_test, y_test, SCALER)

    writer = csv.writer(open('testscaled_predictionsSMALL.csv', mode='w'), delimiter=',', lineterminator='\n')
    writer.writerow(y_test_scaled)
    writer.writerow(predictions)

    print('PLOTING...')
    # Print prediction and y_test
    plt.figure(num=None, figsize=(16, 9))
    test_plot_line, = plt.plot(y_test_scaled[:], color='Blue', label='test data')
    prediction_plot_line, = plt.plot(predictions[:], color='Green', label='prediction')
    plt.legend(handles=[test_plot_line, prediction_plot_line])
    plt.ylabel('USD')
    plt.xlabel('days')
    plt.title('input_seq_length=' + str(input_seq_length) + '|number_units=' + str(number_units) + '|number_epochs=' + str(
            number_epochs) + '|batch_size=' + str(batch_size) + '|RMSE=' + str(rmse))
    plt.savefig(IMG_FOLDER+model_filename+'.png')
    plt.clf()

    # Print a zoomed figure of the prediction and y_test
    zoom_lower_bound = 200
    zoom_upper_bound = 300
    plt.figure(num=None, figsize=(16, 9))
    test_plot_line_zoom, = plt.plot(y_test_scaled[zoom_lower_bound:zoom_upper_bound], color='Blue', label='test data')
    prediction_plot_line_zoom, = plt.plot(predictions[zoom_lower_bound:zoom_upper_bound], color='Green',
                                          label='prediction')
    plt.legend(handles=[test_plot_line_zoom, prediction_plot_line_zoom])
    plt.title(
        'input_seq_length=' + str(input_seq_length) + '|number_units=' + str(number_units) + '|number_epochs=' + str(
            number_epochs) + '|batch_size=' + str(batch_size) + '|zoom over: (' + str(zoom_lower_bound) + ',' +
        str(zoom_upper_bound) + ')')
    plt.ylabel('USD')
    plt.xlabel('days')
    plt.savefig(IMG_FOLDER + model_filename + '_zoomed.png')
    plt.clf()

    return rmse


if __name__ == "__main__":

    # The grid defined can consist of several values per hyperparameter:
    # grid = {
    #     "input_seq_lenth": [5, 10, 15, 20, 60, 80, 100, 120, 150],
    #     "number_units": [50, 100, 150, 200, 500],
    #     "number_epochs": [3, 5, 8, 15, 25, 50],
    #     "batch_size": [32, 64, 96, 128]
    # }

    grid = {
        "input_seq_lenth": [5],
        "number_units": [500],
        "number_epochs": [3],
        "batch_size": [1]
    }

    # Grid search
    for params in ParameterGrid(grid):
        print(params)
        rmse= calculate_rmse(params['input_seq_lenth'], params['number_units'],
                             params['number_epochs'], params['batch_size']),
        print(rmse)
