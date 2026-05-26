import yfinance as yf
import pandas as pd
import numpy as np
import math
import cmath
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

##########Project description##########
#We are implementing an Heston model for pricing European options via FFT (Fast Fourier Transform)
#The pricer is used for calibrating the Heston parameters to provide the best calibration perf against observable option prices

# Acknowledgements & Credits:
# The mathematical core of the characteristic function (`generic_Heston`) and the Fourier 
# transform logic (`genericFFT`) are adapted from the Carr-Madan framework as presented in 
# Coursera's curriculum "Financial Engineering and Risk Management - Columbia University". 
# Optimization infrastructure, data ingestion pipeline, and post-mortem analysis built by the author

########## Step 1 - Data Ingestion (Options) ##########
#Nothing complicated, we are just importing option prices - we will be using call options onlhy for simplicity

ticker_symbol = "TSLA"
tsla = yf.Ticker(ticker_symbol)

#Get current stock price (S0) - Will be used later on
s0 = tsla.fast_info['lastPrice']
print(f"Current Stock Price (S0) for {ticker_symbol}: ${s0:.2f}\n")

#See all available expiration dates
expirations = tsla.options
print(f"Found {len(expirations)} available expiration dates.")
print(f"First few dates: {expirations[:5]}\n")

#Let's pull data for just one expiration date as a test
target_date = expirations[5]
opt_chain = tsla.option_chain(target_date)
target_date

#Extract Call options and keep only what Heston needs
df_calls = opt_chain.calls[['strike', 'bid', 'ask','lastPrice']]
# Create a 'Mid' price
df_calls['Mid'] = df_calls[['bid', 'ask']].mean(axis=1)
#Use the last price as fallback for mid when mid is 0
df_calls['Mid'] = np.where(df_calls['Mid'] == 0.0, df_calls['lastPrice'], df_calls['Mid'])
# Filter out rows where the option price is STILL 0 (completely dead strikes)
df_calls = df_calls[df_calls['Mid'] > 0.0]
# Add a column for maturity
df_calls['Maturity_Date'] = target_date
df_calls.head()

########## Step 2 - Heston characterstic function ##########
#The fancy bit; this is a function that computes the Heston characteristic function
#The function takes as input the options characteristics (stock price S0, interest rates r,time to maturity T) and the Heston parameters (see next)
#Heston parameters: kappa mean reversion speed, theta long term price variance, sigma vol of vol, rho  correlation between stock price and variance returns, and V0 variance today 
#There is an extra input u that represents the wave frequency, represented by a complex number
#The output is phi - a complex number that represents the asset future price distribution
def generic_Heston(u, kappa,theta,sigma,rho, v0,S0, r,d, T):
                
    tmp = (kappa-1j*rho*sigma*u)
    g = np.sqrt((sigma**2)*(u**2+1j*u)+tmp**2)
        
    pow1 = 2*kappa*theta/(sigma**2)
        
    numer1 = (kappa*theta*T*tmp)/(sigma**2) + 1j*u*T*(r-d) + 1j*u*math.log(S0) #Note how the div yield is included as r-d
    log_denum1 = pow1 * np.log(np.cosh(g*T/2)+(tmp/g)*np.sinh(g*T/2))
    tmp2 = ((u*u+1j*u)*v0)/(g/np.tanh(g*T/2)+tmp)
    log_phi = numer1 - log_denum1 - tmp2
    phi = np.exp(log_phi)
            
    return phi

########## Step 3 - FFT vanilla option pricer ##########
#Produces an output of option prices for a strike prices grid
def genericFFT(kappa,theta,sigma,rho,V0,s0,r,d,T,alpha,eta,n):
    #Grid size - the larger the more accurate
    N = 2**n 
    #
    #step-size in log strike space
    #lambda represents the distance between 2 strikes in the FFT grid
    lda = (2*np.pi/N)/eta
    #
    #Choice of beta
    #beta represents the FFT grid starting point
    beta = np.log(s0) - (N * lda) / 2
    #
    #Initiate arrays for the strikes km and for the proba waves Xx
    km = np.zeros((N))
    xX = np.zeros(N, dtype=complex)
    #
    # discount factor
    df = math.exp(-r*T)
    #
    frequency = np.arange(N)*eta
    #we call the Heston function to get a grid of complex wave positions, and then divide that by the Carr-Madan modifier that translates the raw proba into call option payoff shapes
    psi_frequency = generic_Heston(frequency-(alpha+1)*1j,kappa,theta,sigma,rho,V0,s0,r,d,T)/((alpha + 1j*frequency)*(alpha+1+1j*frequency))
    #
    #loop through the FFT grid, allocate the strike price at each point, and then give it its proba wave from our heston function
    #the if=0 condition is related to the trapezoidal rule - because the first bar at starting edge j=0 only has a curve on its right
    for j in range(N):  
        km[j] = beta+j*lda
        if j == 0:
            wJ = (eta/2)
        else:
            wJ = eta
        xX[j] = cmath.exp(-1j*beta*frequency[j])*df*psi_frequency[j]*wJ
    #
    #yY is an array of complex number but does not get yet an option price
    yY = np.fft.fft(xX)
    #
    #to get option cash price cT, we need to remove the imaginary part, and to undo the damping factor alpha
    cT_km = np.zeros((N))  
    for i in range(N):
        multiplier = math.exp(-alpha*km[i])/math.pi
        cT_km[i] = multiplier*np.real(yY[i])
    #
    return km, cT_km

########## Step 4 - Testing the functions ##########
#FFT parameters
alpha=1.5 #damping factor - stability factor
eta=0.25 #spacing between integration slices
n = 12 #factor driving the grid size

#Heston parametes
kappa=2 #mean reversion speed
theta=0.05 #long term price variance
sigma=0.2 #vol of vol
rho=0.5 #correlation stock price - variance return
V0=0.2 #variance today

#Stock parameter - s0 the current stock price was defined earlier in step 1
d=0 #dividend yield

#Option time parameters
#Note - we are using the target date defined in step 1
# Calculate actual fractional T dynamically
today = datetime.now().date()
expiry_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
days_to_maturity = (expiry_dt - today).days
days_to_maturity

# T is measured as a fraction of a 365-day year
T = max(days_to_maturity / 365.0, 0.001) # prevent dividing by zero if expiring today
print(f"Adjusted Option Maturity (T): {T:.5f} years ({days_to_maturity} days)")

#Interest rates
r=0.05

# Generate the whole surface grid
log_strikes, prices = genericFFT(kappa, theta, sigma, rho, V0, s0, r, d, T, alpha, eta, n)
dollar_strikes = np.exp(log_strikes)

# Let's find the price for a specific Target Strike, e.g., K = 180
target_K = 180.0
# Find the closest strike in our generated grid
closest_idx = np.abs(dollar_strikes - target_K).argmin()

# --- YAHOO FINANCE MARKET OUTPUT ---
# Extract the underlying arrays from Step 1 dataframe
market_strikes = df_calls['strike'].values
market_mid_prices = df_calls['Mid'].values

# Find the index of the row in df_calls that is closest to our target strike
closest_market_idx = np.abs(market_strikes - target_K).argmin()

# Pull the exact strike and price Yahoo Finance is reporting
actual_market_strike = market_strikes[closest_market_idx]
actual_market_price = market_mid_prices[closest_market_idx]

print(f"--- HESTON PRICER OUTPUT (pre parameters tuning) ---")
print(f"Closest Strike found in grid: ${dollar_strikes[closest_idx]:.2f}")
print(f"Calculated Heston Call Price: ${prices[closest_idx]:.2f}")
print(f"--- REAL MARKET DATA OUTPUT ---")
print(f"Closest Strike found in Yahoo Data: ${actual_market_strike:.2f}")
print(f"Actual Market Option Price (Mid): ${actual_market_price:.2f}\n")

########## Step 5 - Calibration ##########
#The last step is to calibrate the heston parameters to provide the best fit to observable prices
#We will use the RMSE as objective function (Root Mean Square Error)
def heston_objective_function(parameters):
    # Unpack the parameters the optimizer is testing
    kappa, theta, sigma, rho, V0 = parameters
    
    # Parameter Safeguards (Feller Condition & Boundaries)
    # rho must be between -1 and 1, variances/speeds must be positive
    if not (-1.0 <= rho <= 1.0) or kappa <= 0 or theta <= 0 or sigma <= 0 or V0 <= 0:
        return 1e10  # Return a massive penalty score if parameters are physically impossible
        
    #Run FFT engine using the parameters under trial
    log_strikes, fft_prices = genericFFT(kappa, theta, sigma, rho, V0, s0, r, d, T, alpha, eta, n)
    dollar_strikes = np.exp(log_strikes)
    
    #Interpolate the model prices to line up exactly with market strikes
    #(Since market strikes are discrete e.g., 180, 185, but FFT strikes are continuous decimal grids)
    price_interp = interp1d(dollar_strikes, fft_prices, bounds_error=False, fill_value=0)
    model_prices = price_interp(market_strikes)
    
    #Calculate Total Squared Error
    #We penalize the model heavily for missing liquid, high-volume options
    rmse = np.sqrt(np.mean((market_mid_prices - model_prices) ** 2))
    return rmse

# Initial guesses
initial_guess = [2.0, 0.05, 0.2, 0.5, 0.2]

# --- RUN THE CALIBRATION ---
print("Calibrating Heston model to market data (this may take a few seconds)...")
result = minimize(heston_objective_function, initial_guess, method='Nelder-Mead')

# Extract final parameters
calibrated_kappa, calibrated_theta, calibrated_sigma, calibrated_rho, calibrated_V0 = result.x
final_rmse = result.fun # .fun pulls the final minimized value of the objective function

# --- PRINT THE CALIBRATION REPORT ---
print("             HESTON CALIBRATION REPORT            ")
print(f"Optimization Status : {'SUCCESS' if result.success else 'FAILED'}")
print(f"Termination Reason  : {result.message}")
print(f"Iterations executed : {result.nit}")
print("-"*50)
print(f"FINAL MINIMIZED RMSE: ${final_rmse:.4f}")
print("-"*50)
print("CALIBRATED HESTON PARAMETERS:")
print(f"  Speed of Reversion (kappa) : {calibrated_kappa:.4f}")
print(f"  Long-term Variance (theta) : {calibrated_theta:.4f}")
print(f"  Volatility of Vol (sigma)  : {calibrated_sigma:.4f}")
print(f"  Leverage / Correlation(rho): {calibrated_rho:.4f}")
print(f"  Variance Today (V0)        : {calibrated_V0:.4f}")
print("="*50)

##########Step 6 - Plotting##########

#Generate the final calibrated option prices from tuned parameters
log_strikes_calib, fft_prices_calib = genericFFT(
    calibrated_kappa, calibrated_theta, calibrated_sigma, calibrated_rho, calibrated_V0, 
    s0, r, d, T, alpha, eta, n
)
dollar_strikes_calib = np.exp(log_strikes_calib)

#Map the continuous FFT calibrated grid back to the discrete market strikes
calib_interp = interp1d(dollar_strikes_calib, fft_prices_calib, bounds_error=False, fill_value=0)
calibrated_model_prices = calib_interp(market_strikes)

#Create the plot
plt.figure(figsize=(10, 6))

#Plot real market prices as dots
plt.scatter(market_strikes, market_mid_prices, color='red', alpha=0.6, label='Observable Market Prices (Yahoo)')

#Plot calibrated Heston model prices as a smooth line
plt.plot(market_strikes, calibrated_model_prices, color='navy', linewidth=2, label=f'Calibrated Heston (RMSE: ${final_rmse:.2f})')

#Add context lines (e.g., highlighting where the current stock price sits)
plt.axvline(x=s0, color='gray', linestyle='--', alpha=0.7, label=f'Spot Price (S0 = ${s0:.2f})')

# Formatting details
plt.title(f"Heston Model Calibration Fit for {ticker_symbol} (Expiry: {target_date})", fontsize=14, fontweight='bold')
plt.xlabel("Strike Price ($)", fontsize=12)
plt.ylabel("Call Option Price ($)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

# Display the final visualization
plt.show()

##########Comments##########
#What this script can be used for: to calibrate the Heston 5 parameters to provide the best fit to observable european vanilla options
#What this script cannot be used for: to price/value exotic options
#Why: the maths in FFT technique here only works for vanilla options - notably because it only looks at the asset distribution at time T
#To move into exo options pricing territory, we would need to go into Monte Carlo or PDE solvers techniques instead of FFT
#Monte Carlo usually best handling path dependent and high dimension payoffs, PDE usually best handling early redemption features
#Comments on the goodness of the calibration fit: the code here shows a very good fit for at the money and in the money options, with a relatively poorer fit for out the money options
#This is a known limitation of heston model that struggles in handling deep OTM strikes especially for shorter term maturities
#That is because the model assumes a relatively smooth volatility diffusion process which does not match reality where jumps or regime changes can happen
#That weakness can be adressed for e.g. by transitioning towards jump processes, or introducing two heston diffusions, or potentially can be ignored for e.g. if the trading desk does not trade in short term OTM options