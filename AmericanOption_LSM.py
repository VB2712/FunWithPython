import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#======================================##
#American option pricer using simulations
#We will be working with LSM algo (Longstaff-Schwartz)
#Starting on a Call option and with non dividend paying underlying to simplify
#We expect convergence to the BS price due to non-dividend paying assumption
#======================================#

#======================================##
#Step 1 - Parameters
#======================================##

V=0.2 #volatility - annualized
S=100 #current underlying price
K=100 #strike
R=0.01 #near risk free rate - annualized
T=2 #time to maturity in years
t_increment=10 #time step increment in days in the simulation

#======================================##
#Step 1 - BS pricer
#Will only be used for benchmarking purpose
#======================================##

#BS function for European call on non-distributing stock
def CallBS(S,K,r,T,sigma):
    d1=(np.log(S/K)+(r+np.power(sigma,2)/2)*T)/sigma*np.sqrt(T)
    d2=d1-sigma*np.sqrt(T)
    return norm.cdf(d1)*S-norm.cdf(d2)*K*np.exp(-r * T)

#======================================##
#Step 2 - MC simulation
#Simulate the underlying asset price
#Note that the MC is vectorized - much more efficient than loop
#======================================##

def MonteCarlo(runs,T,S,sigma,r,time_increment,K):

    #number of steps per run and time change per step
    steps=int(T*365/time_increment)
    dt=T/steps

    # Generate all random numbers (Z) in one go
    Z = np.random.standard_normal((runs, steps))

    # Calculate the exponent term for all paths and all steps
    drift_term = (r - 0.5 * sigma**2) * dt
    volatility_term = sigma * np.sqrt(dt) * Z
    
    # Calculate log returns
    log_returns = drift_term + volatility_term
    
    # Use np.cumsum to create the final price paths
    # S_t = S_0 * exp(sum of log returns)
    stock_price = S * np.exp(np.cumsum(log_returns, axis=1))
    
    # Calculate intrinsic value
    intrinsic_value = np.maximum(stock_price - K, 0)
    
    #Return the full array of stock price paths and the corresponding intrinsinc value
    return stock_price, intrinsic_value

#======================================##
#Step 3 - LSM algo - Least square regression
#======================================##

def LSM(runs,S,T,sigma,r,time_increment,K): 

    #Define number of steps per run
    steps=int(T*365/time_increment)
    dt = T/steps # Time step in years

    #Call the Monte Carlo function
    stock_price_path, intrinsic_value_path = MonteCarlo(runs,T,S,sigma,r,time_increment,K)

    # This array will hold the actual discounted cash flow for each path
    V = intrinsic_value_path[:, steps - 1] 

    #Compute the discount factor
    discount_factor = np.exp(-r * dt)

    #LSM core algo - Working backward    
    for j in reversed(range(steps-1)):
        
        #Create a boolean to identify ITM paths
        where_ITM = intrinsic_value_path[:, j] > 0

        if np.any(where_ITM): #we only run the regression if there is at least 1 node ITM to avoid error
            
            #Perform the linear regresionn
            X = stock_price_path[where_ITM,j].reshape(-1,1)
            Y = V[where_ITM] * discount_factor 
            reg = LinearRegression().fit(X, Y)
            continuation_value=reg.predict(X)
            intrinsic_value_ITM=intrinsic_value_path[where_ITM,j]
            
            #Boolean to store the early exercise decision
            where_exercise = intrinsic_value_ITM >= continuation_value

            #Update V for paths where exercise is optimal
            V[where_ITM][where_exercise] = intrinsic_value_ITM[where_exercise]
            
            #print(continuation_value, intrinsic_value_ITM) only used for debugging

        V = V * discount_factor 
    
    #Final option price computed as the average of V across all runs
    option_price = np.mean(V)

    return stock_price_path, intrinsic_value_path,V, option_price


#Only used for debugging
#LSM(5,S,T,V,R,t_increment,K)
#CallBS(S,K,R,T,V)

#======================================##
#Step 4 - Convergence and benchmarking
#======================================##

#convergence parameters and initiate arrays
runs_to_test = [5,10,100,1000,5000,10000,100000]
max_runs=np.max(runs_to_test) #used for plotting purposes
LSM_prices=[] #create an array to store the runs results
stock_paths=[]

#loop to unpack the tupples
for num_runs in runs_to_test:
    stock_path, _, _, LSM_price = LSM(num_runs,S,T,V,R,t_increment,K)
    LSM_prices.append(LSM_price)
    stock_paths.append(stock_path)

#Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

#Plot 1: Sample of Simulated Stock Price Paths. Not that we use the last, largest run to pick the paths from
max_runs_to_plot = 25 #how many paths do we allow to plot at the max
for i in range(min(max_runs_to_plot, max_runs)):
    ax1.plot(stock_paths[-1][i, :]) #Note the way to index the stock price - we use -1 to only call the last run i.e. with the largest amount of runs
ax1.set_xlabel('Days')
ax1.set_ylabel('Stock Price ($)')
ax1.set_title(f'Sample of Simulated Stock Price Paths ({max_runs} runs)')
ax1.grid(True)

# Plot 2: Monte Carlo Convergence
bs_price=CallBS(S,K,R,T,V)
ax2.plot(runs_to_test, LSM_prices, 'o-', label='Monte Carlo Price')
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
#Comments
#======================================##
#We expect the price to converge because we are working with non dividend paying underlying
#There are convergence issues as time to maturity becomes small. Potentially could be improved by using more advanced regression techniques
#A few optimization techniques were used. Notably, using a vectorization instead of loop for the MC function, and also using a time step greater than 1 day (which is also more realistic as in reality exercise decision is not daily)