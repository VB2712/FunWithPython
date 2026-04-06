import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#======================================##
#Basic european option pricer, using Monte Carlo simulation
#Considering a Call option on a stock that does not pay any divident
#First step is to create a BS pricer. This will be used to test convergence of the MC simulations
#Then we will implement the actual MC pricing
#And finally we will implement a convergence back test
#======================================#

#======================================##
#Step 1 - Parameters
#======================================##

sigma=0.2 #volatility - annualized
S=100 #current underlying price
K=100 #strike
r=0.01 #near risk free rate - annualized
T=1 #time to maturity in years

#======================================##
#Step 1 - BS pricer
#======================================##

#BS function for European call on non-distributing stock
def CallBS(S,K,r,T,sigma):
    d1=(np.log(S/K)+(r+np.power(sigma,2)/2)*T)/sigma*np.sqrt(T)
    d2=d1-sigma*np.sqrt(T)
    return norm.cdf(d1)*S-norm.cdf(d2)*K*np.exp(-r * T)

#======================================##
#Step 2 - MC simulation
#======================================##

def MonteCarlo(runs,T,S,K):

    time_increment=1 #time step in days

    #number of steps per run and time change per step
    steps=int(T*365/time_increment)
    dt=time_increment/365

    #Initialize arrays to store results
    stock_price_run=np.zeros((runs,steps)) #stock price for each step of each run
    stock_final_price_run=np.zeros(runs) #final stock price for each run
    option_final_payoff_run=np.zeros(runs) #payoff for each run
    option_value_run=np.zeros(runs) #option PV for each run

    #MC loop to generate stock price
    for i in range(runs):
        stock_price=S

        for j in range(steps):
            # Generate a random number from a standard normal distribution
            random_shock = np.random.normal(0, 1)

            # Apply the Geometric Brownian Motion formula
            # S_t+dt = S_t * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
            drift = (r - 0.5 * np.power(sigma,2)) * dt #note the variance adjustment factor to the drift term - required by Ito's lemma
            volatility = sigma * np.sqrt(dt) * random_shock

            stock_price *= np.exp(drift + volatility)
            stock_price_run[i, j] = stock_price
            
        # Store the final simulated price for this run, as well as the option payoff and payoff value discounted to today (PV of the option)
        stock_final_price_run[i] = stock_price
        option_final_payoff_run[i]=max(0,stock_price-K)
        option_value_run[i]=option_final_payoff_run[i]*np.exp(-r * T)

    #Calculate the PV of the option
    option_pv=np.mean(option_value_run)

    #Return both the option value and the full array of stock price paths
    return option_pv, stock_price_run

#======================================##
#Step 3 - Convergence & Plotting
#======================================##

#convergence parameters
runs_to_test = [5,10,20,40,80,160,320,640,1280,2560]
max_runs=np.max(runs_to_test) #used for plotting purposes
mc_prices=[] #create an array to store the runs results
stock_paths=[]

#loop to unpack the MC tupples
for num_runs in runs_to_test:
    mc_price,stock_path = MonteCarlo(num_runs,T,S,K)
    mc_prices.append(mc_price)
    stock_paths.append(stock_path)

#Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

#Plot 1: Sample of Simulated Stock Price Paths. Not that we use the last, largest run to pick the paths from
max_runs_to_plot = 100 #how many paths do we allow to plot at the max
for i in range(min(max_runs_to_plot, max_runs)):
    ax1.plot(stock_paths[-1][i, :]) #Note the way to index the stock price - we use -1 to only call the last run i.e. with the largest amount of runs
ax1.set_xlabel('Days')
ax1.set_ylabel('Stock Price ($)')
ax1.set_title(f'Sample of Simulated Stock Price Paths ({max_runs} runs)')
ax1.grid(True)

# Plot 2: Monte Carlo Convergence
bs_price=CallBS(S,K,r,T,sigma)
ax2.plot(runs_to_test, mc_prices, 'o-', label='Monte Carlo Price')
ax2.axhline(y=bs_price, color='r', linestyle='--', label='Black-Scholes Price')
ax2.set_xlabel('Number of Simulation Runs')
ax2.set_ylabel('Option Price')
ax2.set_title('Monte Carlo Convergence to Black-Scholes Price')
ax2.legend()
ax2.grid(True)

# Ensure proper spacing between subplots
plt.tight_layout()

plt.savefig('combined_plot.png')
# Save the final combined figure
plt.show()

#======================================##
#Step 3 - Comments & Observations
#======================================##
#There are multiple ways to speed up convergence, for e.g. Anthitetic whereby for each random shock we also take its reverse