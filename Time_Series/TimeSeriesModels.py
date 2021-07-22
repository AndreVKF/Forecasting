## IMPORTS ##
import numpy as np
import pandas as pd
import math

import sys

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Retrieve and manage data
data = pd.read_csv("..\Datasets\datasets_56102_107707_monthly-beer-production-in-austr.csv", parse_dates=['Month'])
data.set_index("Month", inplace=True)

# Split data between training/test
# prct = 0.8
# dfLen = len(data)

# dataCap = math.floor(dfLen*prct)
# LTM
train_data = data[:len(data)-12]
test_data = data[len(data)-12:]

result_data = test_data

# Data chart
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=data.index
            ,y=data['Monthly beer production']
            ,mode='lines')
);
fig.update_layout(title="Monthly beer production Australia")
fig.show()

##### SARIMAX (Autoregressive Integrated Moving Average) #####

# S => Seasonal
# AR => Auto regressive
# MA => Moving Average
# X => Exogenous

## Arguments => (p, d, q), (P, D, Q, m)
# p => number of lag observations (lag order)
# d => number of times the raw observation is differenced (degree of differencing)
# q => size of mov. window (average window)

# P => Seasonal autoregressive order
# D => Seasonal difference order
# Q => Seasonal mov. avg. order
# m => Number of steps for each seasonal period
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

# Decompose chart
chart_SeasonalDecompose = seasonal_decompose(data["Monthly beer production"], model="add")
chart_SeasonalDecompose.plot();

# Sasonalidade
chart_SeasonalDecompose.seasonal[-12*10:].plot()

# Auto ARIMA for best fit
# m => Period for seasonal differencing (4 => Quarter, 12 => Monthly, 1 => Annual)
auto_arima(data['Monthly beer production'], seasonal=True, m=12, max_p=7, max_d=5, max_q=7, max_P=4, max_D=4, max_Q=4).summary()

# Create fit/model & predict
sarimax_model = SARIMAX(train_data['Monthly beer production'], order=(5, 1, 4), seasonal_order=(1, 0, 1, 12))
sarimax_fit = sarimax_model.fit()
sarimax_predict = sarimax_fit.predict(start=test_data.index[0], end=test_data.index[-1], typ="levels")

# Add SARIMAX prediction to result DF
result_data['SARIMAX'] = sarimax_predict

# Error Valuation
sarimax_rmse_error = rmse(test_data["Monthly beer production"], sarimax_predict)
sarimax_mse_error = sarimax_rmse_error ** 2

##### LSTM Neural Network #####
# LSTM => Long short term memory
# Mitigates the vanishing gradient problem (gradient weighting issue)
# Input Gate => Scales input to cell (write)
# Output Gate => Scales output to cell (read)
# Forget Gate => Scales od cell value (reset)
# Gates control the flow of the read/write, incorporatign long-term memory function into the model

from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Scale data
scaler = MinMaxScaler()

scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

# Create time series generator
n_input = 12
n_features = 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)

# Create/Train/Fit model
lstm_model = Sequential()
lstm_model.add(LSTM(200, activation="relu", input_shape=(n_input, n_features)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()

lstm_model.fit_generator(generator, epochs=20)

# Convergence test
losses_lstm = lstm_model.history.history['loss']

fig_3 = go.Figure()

fig_3.add_trace(
    go.Scatter(y=losses_lstm)
)

fig_3.update_layout(
    title="Loss Improve by Epoch"
    ,xaxis_title="Epochs"
    ,yaxis_title="Loss")
fig_3.show();

# Predictions
lstm_predictions_scaled = []

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred)
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]], axis=1)

# Invert scale predictions
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

# Add lstm prediction to result DF
result_data['LSTM'] = lstm_predictions

# Error valuation
lstm_rmse_error = rmse(test_data['Monthly beer production'], result_data['LSTM'])
lstm_mse_error = lstm_rmse_error ** 2

##### Chart w/ Predictions #####
fig_2 = go.Figure()

fig_2.add_trace(
    go.Scatter(
        x=train_data.index
        ,y=train_data["Monthly beer production"]
        ,mode='lines'
        ,line_color='#1f77b4'
        ,name="Train Data"
    )
)

fig_2.add_trace(
    go.Scatter(
        x=test_data.index
        ,y=test_data['Monthly beer production']
        ,mode="lines"
        ,line_color='#2ca02c'
        ,name="Test Data"
    )
)

fig_2.add_trace(
    go.Scatter(
        x=result_data.index
        ,y=result_data['LSTM']
        ,mode="lines"
        ,line=dict(
            color='#e377c2'
            ,dash="dot"
        )
        ,name="LSTM Prediction"
    )
)

fig_2.add_trace(
    go.Scatter(
        x=result_data.index
        ,y=result_data['SARIMAX']
        ,mode="lines"
        ,line=dict(
            color='#ff7f0e'
            ,dash="dot"
        )
        ,name="SARIMAX Prediction"
    )
)

fig_2.update_layout(title="Australia Monthly Beer Production Predictions"
                    ,xaxis_title="Date"
                    ,yaxis_title="Volume (ML)")

fig_2.show();



