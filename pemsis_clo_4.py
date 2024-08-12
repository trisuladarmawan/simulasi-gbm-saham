import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import ta
import streamlit as st
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


scaler = MinMaxScaler()

# Data Loading and Preprocessing
def load_data(ticker, start, end, period):
    data = yf.download(ticker, start, end)
    data['return'] = data['Adj Close'].pct_change(period).dropna()
    data['sma'] = ta.trend.sma_indicator(data['Adj Close'], window=period)
    data['ema'] = ta.trend.ema_indicator(data['Adj Close'], window=period)
    data['rsi'] = ta.momentum.rsi(data['Adj Close'], window=period)
    data = data.dropna()
    features = ['sma', 'ema', 'rsi']
    data[features] = scaler.fit_transform(data[features])
    return data, features

def resample_data(data, interval):
    resampled_data = data.resample(interval).last()
    resampled_data['return'] = resampled_data['Adj Close'].pct_change().dropna()
    resampled_data['sma'] = ta.trend.sma_indicator(resampled_data['Adj Close'], window=period)
    resampled_data['ema'] = ta.trend.ema_indicator(resampled_data['Adj Close'], window=period)
    resampled_data['rsi'] = ta.momentum.rsi(resampled_data['Adj Close'], window=period)
    resampled_data = resampled_data.dropna()
    features = ['sma', 'ema', 'rsi']
    resampled_data[features] = scaler.fit_transform(resampled_data[features])
    return resampled_data, features

def train_model(data, features, step):
    X = data[features].iloc[:step].values
    y_drift = data["return"].iloc[:step].values
    y_volatility = data["return"].rolling(window=20).std().iloc[:step].dropna().values
    X_volatility = X[len(X) - len(y_volatility):]
    model_drift = RandomForestRegressor()
    model_drift.fit(X, y_drift)
    model_volatility = RandomForestRegressor()
    model_volatility.fit(X_volatility, y_volatility)


    # Prediksi menggunakan model drift
    y_drift_pred = model_drift.predict(X)

    # Evaluasi menggunakan metrik MAE
    mae_drift = mean_absolute_error(y_drift, y_drift_pred)

    # Evaluasi menggunakan metrik MSE
    mse_drift = mean_squared_error(y_drift, y_drift_pred)

    # Evaluasi menggunakan metrik R^2
    r2_drift = r2_score(y_drift, y_drift_pred)

    return model_drift, model_volatility

def evaluate_model(data, model_drift, features, step):
    X = data[features].iloc[:step].values
    y_drift = data["return"].iloc[:step].values
    y_volatility = data["return"].rolling(window=20).std().iloc[:step].dropna().values
    X_volatility = X[len(X) - len(y_volatility):]

    # Prediksi menggunakan model drift
    y_drift_pred = model_drift.predict(X)

    # Evaluasi menggunakan metrik MAE
    mae_drift = mean_absolute_error(y_drift, y_drift_pred)

    # Evaluasi menggunakan metrik MSE
    mse_drift = mean_squared_error(y_drift, y_drift_pred)

    # Evaluasi menggunakan metrik R^2
    r2_drift = r2_score(y_drift, y_drift_pred)

    
    return mae_drift, mse_drift, r2_drift



def gbm_sim(spot_price, time_horizon, steps, model_drift, model_volatility, features, data, n_sims=100):
    dt = 1
    drift = model_drift.predict(scaler.fit_transform(data[features]))
    volatility = model_volatility.predict(scaler.fit_transform(data[features]))
    all_paths = []
    for _ in range(n_sims):
        prices = [spot_price]
        for i in range(len(data)):
            eps = np.random.normal(scale=np.sqrt(dt))
            S_t = prices[-1] * np.exp((drift[i] - 0.5 * volatility[i]**2) * dt + volatility[i] * eps)
            prices.append(S_t)
        all_paths.append(prices)
    return np.array(all_paths), drift, volatility

# Monte Carlo Simulation
def monte_carlo_simulation(spot_price, time_horizon, steps, model_drift, model_volatility, features, data, n_simulations=1000):
    simulations = []
    for _ in range(n_simulations):
        paths, _, _ = gbm_sim(spot_price, time_horizon, steps, model_drift, model_volatility, features, data, n_sims=1)
        simulations.append(paths)
    simulations = np.array(simulations)
    return simulations

st.title('Stock Price Prediction using GBM and Machine Learning')
st.write("This application simulates future stock prices using Geometric Brownian Motion (GBM) with drift and volatility predicted by machine learning models.")

ticker = st.text_input('Enter stock ticker', 'GGRM.JK')
start_date = st.date_input('Start date', datetime(2019, 1, 1))
end_date = st.date_input('End date', datetime(2023, 12, 31))
n_sims = st.number_input('Number of simulations', value=100)
period = st.number_input('Period for technical indicators', value=5)
interval = st.selectbox('Select prediction interval', ['daily', 'weekly', 'monthly', 'quarterly'])

if st.button('Run Simulation'):
    data, features = load_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), period)

    if interval == 'weekly':
        resampled_data, features = resample_data(data, 'W')
    elif interval == 'monthly':
        resampled_data, features = resample_data(data, 'M')
    elif interval == 'quarterly':
        resampled_data, features = resample_data(data, 'Q')
    else:
        resampled_data = data

    steps = int(len(resampled_data) / 2)
    model_drift, model_volatility = train_model(resampled_data, features, steps)
    spot_price = resampled_data["Adj Close"].iloc[steps-1]
    time_horizon = len(resampled_data) - steps

    # Evaluate Model
    mae_drift, mse_drift, r2_drift = evaluate_model(resampled_data, model_drift, features, steps)
    st.write(f"Mean Absolute Error (Drift): {mae_drift}")
    st.write(f"Mean Squared Error (Drift): {mse_drift}")
    st.write(f"Mean Squared Error (Drift): {r2_drift}")

    # Single GBM Simulation
    simulated_paths, drifts, volatilities = gbm_sim(spot_price, time_horizon, steps, model_drift, model_volatility, features, resampled_data.iloc[steps:], n_sims)

    # Monte Carlo Simulation
    mc_simulations = monte_carlo_simulation(spot_price, time_horizon, steps, model_drift, model_volatility, features, resampled_data.iloc[steps:], n_simulations=n_sims)

    fig, ax = plt.subplots()
    for i in range(n_sims):
        ax.plot(resampled_data.index[steps-1:], mc_simulations[i][0], color='grey', alpha=0.1)
    ax.plot(resampled_data.index[steps-1:], resampled_data['Adj Close'].iloc[steps-1:], color='blue', label='Actual Price')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"Simulated Stock Price Paths ({interval.capitalize()})")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Display the predicted drift and volatility
    st.write('Predicted Drift')
    st.line_chart(drifts)

    st.write('Predicted Volatility')
    st.line_chart(volatilities)

    # Save to CSV
    result = pd.DataFrame({
        'Date': resampled_data.index[steps-1:],
        'Actual': resampled_data['Adj Close'].iloc[steps-1:],
        'Predicted_Mean': np.mean(mc_simulations[:, 0, :], axis=0),
        'Predicted_Std': np.std(mc_simulations[:, 0, :], axis=0)
    })
    csv = result.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f'simulated_stock_prices_{interval}.csv',
        mime='text/csv',
    )
