This project compares long short-term memory neural networks (LSTM) and Facebook Prophet as tools for financial forecasting.

In order to generate new LSTM or Facebook Prophet models, the modules lstm.py or prophet.py respectively should be executed after replacing the existing grid for the desired one, containing the values for the hyperparameters.

The trader simulator loads and executes by default the best models for LSTM and Prophet. In order to change the model to execute, trader_simulator.py should be executed after changing the lstm_model and prophet_model variables inside the code.

The folders LSTM_FINAL_MODELS/, PROPHET_MODELS and LSTM_FIRST_MODELS contain both models and plots of just some of the models generated and tested for the project. The majority of them were not pushed to decrease the project's size.

A requirements.txt file can be found inside venv/ folder in order to install all packages needed:
	pip install -r requirements.txt


