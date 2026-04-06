########## Project Objective ##########
#Objcetive is to ingest some EQ data, process it (labelling and fractioning), XGBOOST it for price prediction purpose, and backtest the results

#Functions import
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########## Step 1 - Data Ingestion ##########
#Import some end of day stock prices with low/high/close/open

# Define the universe
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Download price data from Yahoo finance
data = yf.download(tickers, start="2015-01-01", end="2025-12-31", auto_adjust=True)

# Pulling 5 days of Apple data
df = yf.download("AAPL", period="5d")
print(df[['Open', 'High', 'Low', 'Close', 'Volume']])

# Stack the data into a long table - will make next steps easier
df_long = data.stack(level=1)
df_long.index.names = ['Date', 'Ticker']
df_long = df_long.reset_index()
df_long = df_long.sort_values(['Ticker', 'Date'])
df_long.head()

######### Step 2 - Data Processing #########
#We compute the log return and then a daily volatility using a weighted moving average based on last 100 observations
#Note: for now we keep our dataframe on a tick basis but an alternative is to re based to be volume basis - although less relevant with daily ticks as opposed to intraday ticks

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

######### Step 4 - Fractional Differntiation & Data prep #########
#First, run a test for stationarity - we are using the ADF test to check whether data is suitable for ML or if we need to apply fractional diff
#Using 1 stock as a sample

from statsmodels.tsa.stattools import adfuller

sample_ticker = df_clean[df_clean['Ticker'] == 'AAPL']['Close']
result = adfuller(sample_ticker)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

#As expected, the data is highly non-stationary (p value under 0.05 considered stationary)
#We have to transform it in order to make the data stationary whilst also not completely losing memory

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

#Main loop to find optimal d, using Apple stock as a sample. The loop iterates from 0 to 1 by 0.1 interval
#We set an objective function as the p-value of ADF test to be lower but as close as possible to 0.05
for d in np.arange(0, 1, 0.1):
    series = df_clean[df_clean['Ticker'] == 'AAPL']['Close']
    df_diff = frac_diff_ffd(series, d=d, threshold=1e-4, window=500)
    res = adfuller(df_diff)
    p_val = res[1]
    print(f"Testing d={d:.1f} | p-value: {p_val:.4f}")
    if p_val<=0.05:
        print(f"Optimal d={d:.1f} | p-value: {p_val:.4f}")
        optimal_d=d
        break

#Some comments on fractional differences
#Using d=0 is equivalent to feeding raw prices into our ML models. These prices tend to be non-stationary meaning that their variance and mean changes
#For ML models that means that a pattern working when the stock is worth $50 might not work at $150
#On the other end, using d=1 is equivalent to feeding returns data. Here we eliminate a lot of the time series memory because each day is almsot independent from previous ones
#Using anything in between we are conserving previous prices but with a decreasing weight

####Data preparation for ML model####
#We now need to prepare the data for ML ingestion
#We create a X series with the features - the closing prices post differentiation and we will also add RSI indicator i.e. 2 X variables
#And a Y series with the Labels from previous steps

#Stock: select the stock in ["AAPL", "MSFT", "GOOGL", "AMZN"]
Stock = 'AMZN'

#Create the x variables
df_clean = df_clean.set_index('Date')
#Returns post differentiation
series = df_clean[df_clean['Ticker'] == Stock]['Close']
df_X = frac_diff_ffd(series, d=optimal_d)
df_X.head()
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

#merge both x series
df_X = df_X[~df_X.index.duplicated(keep='first')]
df_rsi_diff = df_rsi_diff[~df_rsi_diff.index.duplicated(keep='first')]
X_df = pd.concat([df_X, df_rsi_diff], axis=1, join='inner').dropna()
X_df.columns = ['FracDiff_Price', 'FracDiff_RSI']
X_df.head()
df_y = df_clean[df_clean['Ticker'] == Stock]['Label']
final_df = pd.concat([X_df, df_y], axis=1, join='inner').dropna()
final_df.head()
X = final_df.iloc[:, :-1].values
y = final_df['Label'].values
print(X[:5])
print(y[:5])

######### Step 5 - Machine Learning #########
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Train the XGBOOST model
split=0.8
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=(1-split), shuffle=False)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
clf.fit(X_train,y_train)

#Print classification report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

#Print confusion matrix
target_labels = [-1, 0, 1]
target_names = ['Sell (-1)', 'Neutral (0)', 'Buy (1)']
cm = confusion_matrix(y_test, y_pred, labels=target_labels)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
print(cm_df)

#Features importance
importances = clf.feature_importances_
print("--- Feature Importance ---")
print(f"FracDiff Price: {importances[0]:.4f}")
print(f"FracDiff RSI:   {importances[1]:.4f}")

######### Step 6 - Back Testing #########
split_test = int(len(X) * split)
test_dates = final_df.index[split_test:]
results = pd.DataFrame(index=test_dates)
results.head()
results['Log_Return'] = df_clean[df_clean['Ticker'] == Stock].loc[test_dates, 'Log_Returns']
results['Prediction'] = y_pred
results.head()
# Shift the prediction to avoid "Look-Ahead Bias"
# We want: Yesterday's Prediction * Today's Return
results['Strategy_Log_Ret'] = results['Prediction'].shift(1) * results['Log_Return']
results.head()

# 3. Calculate Strategy Returns
# If prediction is 1 (Buy), we get the return. 
# If prediction is -1 (Sell), we get the negative return (shorting).
# If prediction is 0 (Neutral), we get 0.
# Since these are LOG returns, we SUM them first, then use np.exp()
results['Cumu_Market'] = np.exp(results['Log_Return'].cumsum())
results['Cumu_Strategy'] = np.exp(results['Strategy_Log_Ret'].cumsum())
print(results[['Cumu_Market', 'Cumu_Strategy']].tail())

#Comments: the market return represents a buy and hold stragy, whilst the strategy represents the return from our AI based strategy
#Plot the results
# Set the figure size
plt.figure(figsize=(14, 7))

# Plot the Market (Benchmark)
plt.plot(results['Cumu_Market'], label='Market (AAPL Buy & Hold)', color='gray', linestyle='--', alpha=0.6)

# Plot the Strategy
plt.plot(results['Cumu_Strategy'], label='ML Strategy (Price + RSI)', color='blue', linewidth=2)

# Add a horizontal line at 1.0 (The "Break-even" line)
plt.axhline(y=1.0, color='black', linestyle='-', alpha=0.2)

# Formatting
plt.title('Backtest: Cumulative Strategy Returns vs. Market', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Cumulative Wealth ($1 Initial)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.show()

#Final comments
#There are a few bugs to fix:
#1/ Should group RSI df_clean.groupby('Ticker')['Close'].transform(lambda x: calculate_RSI(x)) - to avoid jumps when moving from 1 ticket to another
#2/ We should look for the optimal d on the training set only, currently d looked on the entire data set including testing population
#3/ In backtest, we are using the close on close log return which is not realistic, a close on open would be more realistic  
#4/ Split train test should be changed to a purgedkfold
#5/ Trading stragy: instead of sizing the bets as 1,0,-1 we should size the bet based on the proba of the signal, using clf.predict_proba()
#6/ Add more features (volatility, bid-offer spread, and more macro indicators)
#7/ Hyper parameters optimization
#8/ Calibrate d per stock instead of doing a single calibration