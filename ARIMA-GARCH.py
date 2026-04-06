import pandas as pd
import numpy as np
from arch import arch_model #need to be run in dedicated environment - run commmand: python3 -m venv garch_env and then source garch_env/bin/activate
import matplotlib.pyplot as plt

################ Objective ################
#We will implement some basic econometrics via Python
#We will work with GARCH in order to estimate the volatility of financial time series
#We will work with EUR/GBP FX spot rate. Weekly time series on a 2 year horizon (50 data points). Source: Gemini
###########################################

################ Fixed Parameters ################
train_weight=0.8 #split training vs testing data

################ Market Data Import (FINAL FIX: FORCING NUMERIC INDEX) ################
#The data was already clean so we do not do any specific re-treatment or data cleaning
data = pd.read_csv(r'EURGBP_Spot.csv')
data = data.rename(columns={'Date (End of Week)':'Date','EUR/GBP Spot Rate (GBP per EUR)':'Value'})
data['Log_Return'] = np.log(data['Value']/data['Value'].shift(1))
data.dropna(subset=['Log_Return'], inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.head()
data.tail()

# ################ PLot log returns ################
#For splitting visually between test and train
total_points = len(data['Date'])
split_index = int(total_points * train_weight)
split_date = data['Date'].iloc[split_index]
split_date

#Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Date'], data['Log_Return'])

#Add a vertical line to vizualise the split point
ax.axvline(
    x=split_date,
    color='r',
    linestyle='--',
    label='Training/Testing split'
)

#Fine tune ticks interval and dates rotation
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=8))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate(rotation=45)

#Add legend and labels
ax.legend() 
ax.set_xlabel('Date')
ax.set_ylabel('Log return EUR/GBP')
ax.set_title('EUR/GBP Log Return with Train/Test Split')

plt.show()

################ Create training and testing data frames ################
train_size = int(len(data) * train_weight)
test_size = len(data) - train_size
train_returns = data['Log_Return'][:train_size]
test_returns = data['Log_Return'][train_size:].reset_index(drop=True)
train_returns.head()
test_returns.tail()
test_size
train_size
total_points

################ Fit GARCH(1,1) model and print key stats ################
#Note that these are based on weekly returns so would need to be annualized if used for option pricing
am = arch_model(train_returns, mean='Zero', vol='Garch', p=1, q=1) #we use a 1 day lag for both the actual and forecasted returns
res = am.fit(disp='off')
print(res.summary())
forecasts = res.forecast(horizon=test_size) 

################ Out of sample testing ################
forecast_variance = forecasts.residual_variance.iloc[-1, :]
forecast_variance

comparison_df = pd.DataFrame({
    'Actual_Return': test_returns,
    'Realized_Vol_Pct': np.abs(test_returns),
    'Predicted_Variance': forecast_variance.values,
    'Predicted_Vol_Pct': np.sqrt(forecast_variance).values,
    'Error':  np.abs(test_returns) - np.sqrt(forecast_variance).values
})

comparison_df
