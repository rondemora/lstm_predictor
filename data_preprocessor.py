"""
Constains methods to read and prepare train and test datasets from the original raw data.
"""

import numpy as np
import pandas as pd

VALIDATION_SPLIT = 0.8


def get_dataframe_lstm():
    """
    Reads the dataframe from the CSV and prepares it for the LSTM model to use
    """
    df = pd.read_csv('sp5001962.csv', sep=',')
    df = df.drop('Date', axis=1)
    df = df.drop('Open', axis=1)
    df = df.drop('High', axis=1)
    df = df.drop('Low', axis=1)
    df = df.drop('Adj Close', axis=1)
    df = df.drop('Volume', axis=1)
    return df


def get_dataframe_prophet():
    """
    Reads the dataframe from the CSV and prepares it for the Prophet model to use
    """
    df = pd.read_csv('sp5001962.csv', sep=',')
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
    df = df.drop('Open', axis=1)
    df = df.drop('High', axis=1)
    df = df.drop('Low', axis=1)
    df = df.drop('Adj Close', axis=1)
    df = df.drop('Volume', axis=1)
    df.columns = ['ds', 'y']
    return df


def create_samples(full_sequence, n_timesteps):
    """
    Splits the full data sequence into samples, given the number of time steps
    """
    X = []
    y = []
    for i in range(len(full_sequence) - n_timesteps):
        last_index = i + n_timesteps
        seq_x, seq_y = full_sequence[i:last_index], full_sequence[last_index]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def prepare_data_lstm(input_seq_length, scaler, n_features):
    """
    Generates the train and test sets for an LSTM model
    """
    df = get_dataframe_lstm()

    train_dataset = np.array(df[:int(df.shape[0] * VALIDATION_SPLIT)])
    test_dataset = np.array(df[int(df.shape[0] * VALIDATION_SPLIT):])

    # Fit the scaler with the train data, transform both datasets:
    train_dataset = scaler.fit_transform(train_dataset)
    test_dataset = scaler.transform(test_dataset)
    X_train, y_train = create_samples(train_dataset, input_seq_length)
    X_test, y_test = create_samples(test_dataset, input_seq_length)

    # Input has to be of shape (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_features)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_features)

    return X_train, y_train, X_test, y_test


def prepare_data_prophet(log_data=False):
    """
    Generates the train and test sets for an LSTM model
    """
    df = get_dataframe_prophet()

    if log_data:
        df['y'] = df['y'].apply(lambda x: np.log(x))

    train_dataset = df[:int(df.shape[0] * VALIDATION_SPLIT)]
    test_dataset = df[int(df.shape[0] * VALIDATION_SPLIT):]
    return df, train_dataset, test_dataset
