########## Project Objective ##########
#This project re-uses the design of the XGBOOST script, but instead of using an XGBOOST model we will use an RNN
#More specificially - LSTM design
#Inspired by 'Machine Learning for Algo Trading - S. Jansen'

#Functions import
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########## Step 1 - Data Ingestion ##########
#Import some end of day stock prices with low/high/close/open

# Define the universe of stocks
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Download price data from Yahoo finance
data = yf.download(tickers, start="2015-01-01", end="2025-12-31", auto_adjust=True)

# Stack the data into a long table - will make next steps easier
df_long = data.stack(level=1)
df_long.index.names = ['Date', 'Ticker']
df_long = df_long.reset_index()
df_long = df_long.sort_values(['Ticker', 'Date'])
df_long.head()

######### Step 2 - Data Processing #########
#We compute the log return and daily volatility using a weighted moving average based on last 100 observations

df_long['Log_Returns'] = df_long.groupby('Ticker')['Close'].transform(lambda x: np.log(x).diff())
df_long['Daily_Vol'] = df_long.groupby('Ticker')['Log_Returns'].transform(lambda x: x.ewm(span=100).std())
df_long.head()

######### Step 3 - Triple Barrier Labelling #########
# For each day, we look ahead and check whichever is hit first: time limit, low spot limit, high spot limit

vertical_limit = 5 #how many days can a trade be kept open
sigma_high = 2 #how many standard deviations for the upwards barrier
sigma_low = 2 #how many standard deviations for the downards barrier

# Add the barriers
df_long['Date'] = pd.to_datetime(df_long['Date'])
df_long['Vertical_Barrier'] = df_long['Date'] + pd.Timedelta(days=vertical_limit)
df_long['High_Barrier'] = df_long['Close']*(1+sigma_high*df_long['Daily_Vol'])
df_long['Low_Barrier'] = df_long['Close']*(1-sigma_low*df_long['Daily_Vol'])
df_long.head()

# Function that looks ahead and define whether the high or low horizontal barrier is hit first 
def get_triple_barrier_labels(df_ticker, vertical_limit):
    
    labels = []
    
    for i in range(len(df_ticker) - vertical_limit):
        price_window = df_ticker.iloc[i : i + vertical_limit]
        
        hi_barrier = df_ticker['High_Barrier'].iloc[i]
        lo_barrier = df_ticker['Low_Barrier'].iloc[i]
        
        hi_hit = price_window['High'] >= hi_barrier
        lo_hit = price_window['Low'] <= lo_barrier
        
        hi_hit_idx = hi_hit.idxmax() if hi_hit.any() else None
        lo_hit_idx = lo_hit.idxmax() if lo_hit.any() else None
        
        if hi_hit_idx and (not lo_hit_idx or hi_hit_idx < lo_hit_idx):
            labels.append(1)  # high hit first
        elif lo_hit_idx and (not hi_hit_idx or lo_hit_idx < hi_hit_idx):
            labels.append(-1) # low hit first
        else:
            labels.append(0)  # vertical varrier hit first
            
    # Pad the end with NaNs because we can't look ahead at the very end of the data
    return pd.Series(labels + [np.nan] * vertical_limit, index=df_ticker.index)

# Apply the triple barrier function to create a 1,-1,0 label
df_clean = df_long.dropna(subset=['High_Barrier', 'Low_Barrier']).copy()
df_clean['Label'] = df_clean.groupby('Ticker', group_keys=False).apply(lambda x: get_triple_barrier_labels(x, vertical_limit))
df_clean.head(10)

######### Step 4 - Fractional Differntiation #########
#We create 2 functions which will then be called in a loop in order to compute the weights for a given d and apply the weights to a data series
def get_weights_ffd(d, threshold=1e-5, window=500):
    """
    Generates weights for the Fractional Differentiation.
    d: The order of differentiation (e.g., 0.4)
    threshold: The cutoff for weight significance
    """
    w = [1.0]
    for k in range(1, window):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
        if abs(w_k) < threshold:
            break
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, threshold=1e-5, window=500):
    """
    Applies the weights to the series.
    """
    weights = get_weights_ffd(d, threshold, window)
    width = len(weights) - 1
    
    # We apply the weights across a rolling window
    output = []
    for i in range(width, len(series)):
        # Dot product of the weights and the price window
        dot_prod = np.dot(weights.T, series.iloc[i-width : i+1])[0]
        output.append(dot_prod)
        
    return pd.Series(output, index=series.index[width:])

#Main loop to find optimal d, using one stock as a sample. The loop iterates from 0 to 1 by 0.1 interval
#We set an objective function as the p-value of ADF test to be lower but as close as possible to 0.05

sample_stock='GOOGL'

from statsmodels.tsa.stattools import adfuller

for d in np.arange(0, 1, 0.1):
    series = df_clean[df_clean['Ticker'] == sample_stock]['Close']
    df_diff = frac_diff_ffd(series, d=d, threshold=1e-4, window=500)
    res = adfuller(df_diff)
    p_val = res[1]
    print(f"Testing d={d:.1f} | p-value: {p_val:.4f}")
    if p_val<=0.05:
        print(f"Optimal d={d:.1f} | p-value: {p_val:.4f}")
        optimal_d=d
        break

####Step 5: Data preparation for ML model####
#We now need to prepare the data for ML ingestion
#We create a X series with the features - the closing prices post differentiation and we will also add RSI indicator i.e. 2 X variables
#And a Y series with the Labels from previous steps

#Stock: select the stock in ["AAPL", "MSFT", "GOOGL", "AMZN"]
Stock = 'GOOGL'

#Create the x variables
df_clean = df_clean.set_index('Date')
series = df_clean[df_clean['Ticker'] == Stock]['Close']
df_return = frac_diff_ffd(series, d=optimal_d)
df_return.head()

#Add RSI
def calculate_RSI (series, window=14):
    delta = series.diff()
    gain = (delta.where(delta>0.0))
    loss = (-delta.where(delta<0.0))
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_clean['RSI'] = calculate_RSI(df_clean['Close'], window=14)
series_rsi = df_clean['RSI'].dropna()

#note - we use a lower d than our optimal d - because RSI expected to be more stationnary than price
df_rsi_diff = frac_diff_ffd(series_rsi, d=0.1, threshold=1e-4, window=500)
df_rsi_diff.head()

#merge returns and RSI into a single table
df_return = df_return[~df_return.index.duplicated(keep='first')]
df_rsi_diff = df_rsi_diff[~df_rsi_diff.index.duplicated(keep='first')]
X_df = pd.concat([df_return, df_rsi_diff], axis=1, join='inner').dropna()
X_df.columns = ['FracDiff_Price', 'FracDiff_RSI']
X_df.head()

#extract the y as the labelled stock, and merge with the table created in the previous step
df_y = df_clean[df_clean['Ticker'] == Stock]['Label']
final_df = pd.concat([X_df, df_y], axis=1, join='inner').dropna()
final_df.head()

X = final_df.iloc[:, :-1].values
y = final_df['Label'].values
print(X[:5])
print(y[:5])

##########Step 6 - Prepare the data for RNN##########
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#Scale the data
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)

#Define the Windowing function
def create_rnn_sequences(X_data, y_data, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - window_size):
        X_seq.append(X_data[i : i + window_size])
        y_seq.append(y_data[i + window_size])
    return np.array(X_seq), np.array(y_seq)

# Choose lookback (e.g., use the last 20 days of patterns to predict)
window_size = 20
X_rnn, y_rnn = create_rnn_sequences(X_scaled, y, window_size)

#Train/Test Split
train_split=0.8
split = int(len(X_rnn) * train_split)
X_train, X_test = X_rnn[:split], X_rnn[split:]
y_train, y_test = y_rnn[:split], y_rnn[split:]

######### Step 7 - Machine Learning #########
# Convert labels to categorical (0, 1, 2) for the neural network
# -1 becomes 0, 0 becomes 1, 1 becomes 2
from tensorflow.keras.utils import to_categorical
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# Define Model
model = Sequential()

# First LSTM Layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# Second LSTM Layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Output Layer: 3 units for (-1, 0, 1) and 'softmax' for probabilities
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train_cat, epochs=20, batch_size=32, validation_data=(X_test, y_test_cat))

##########step 8: model diagnostic
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Get the model's probability predictions
y_pred_probs = model.predict(X_test)

#Convert probabilities to class indices (0, 1, 2)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

#Convert categorical test labels back to class indices (0, 1, 2)
y_true_classes = np.argmax(y_test_cat, axis=1)

#Create the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Define the labels for the plot
target_names = ['Low Hit (-1)', 'Vertical (0)', 'High Hit (1)']

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues', ax=ax)

plt.title(f'Confusion Matrix: LSTM Model ({Stock})')
plt.show()

######### Step 9 - Back Testing #########
#Filter df_clean for only the stock we modeled
df_stock_test = df_clean[df_clean['Ticker'] == Stock]

# Get the dates for the test set
test_dates = final_df.index[split + window_size:]

# Get the actual log returns for those dates
test_returns = df_stock_test.loc[test_dates, 'Log_Returns'].values

# Get model signals (-1, 0, 1)
# We map the argmax (0,1,2) back to our trading signals (-1,0,1)
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
strategy_signals = y_pred_classes - 1  # Shifts [0,1,2] to [-1,0,1]

# Calculate daily strategy returns
# We shift signals by 1 day to avoid look-ahead bias (trading on today's close for tomorrow's return)
strategy_returns = strategy_signals[:-1] * test_returns[1:]

# Calculate cumulative returns
cum_strategy_returns = np.exp(np.cumsum(strategy_returns)) - 1
cum_market_returns = np.exp(np.cumsum(test_returns[1:])) - 1

plt.figure(figsize=(12, 6))
plt.plot(test_dates[1:], cum_strategy_returns, label='LSTM Strategy', color='orange')
plt.plot(test_dates[1:], cum_market_returns, label='Market (Buy & Hold)', color='blue', alpha=0.5)
plt.title(f'Backtest Results: {Stock}')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()