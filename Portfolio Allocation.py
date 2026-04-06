########## Portfolio Allocation ##########
#We will import some stock daily returns data, and as an initial step compute the average daily return and the covariance matrix
#We will then run an optimization algo to look for the optimal allocation (max expected return and min expected variance with the available stocks, with and without short)
#Finally we will plot the efficiency frontier using a monte carlo simulation

########## Libraries import ###########
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

########## Step 1 - Import stock daily return ##########
#Note that the data is already cleaned and is in the form of daily returns
data = pd.read_csv('Stock_Returns.csv')
data.head()

########## Step 2 - Compute the average daily return and the covariance matrix ##########
#We first compute the average returns and the covariance matrix and convert both into an array
covariance = data.drop(columns=['date','Avg']).cov()
avg_return=data.drop(columns=['date','Avg']).mean()
avg_return_array=avg_return.values
covariance_array=covariance.values
covariance_array
avg_return_array

########## Step 3 - find optimal portfolio allocation (using Sharpe ratio maximisation) ##########

#First, define the risk free rate  
rfr=0.0001

#Objective function that we want to minimise. Note that we are taking the negative of the Sharpe ratio as we are using a minimisation technique
def objective(weights,covariance,average_return):
    ptf_return=np.sum(average_return * weights)
    ptf_stdev = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
    return - (ptf_return-rfr)/ptf_stdev

#Constraints
def weight_constraint(weights):
    return np.sum(weights) - 1

constraints = ({'type': 'eq', 'fun': weight_constraint}) #'eq' means equality i.e. the function must equal 0

#Bounds
#bounds = tuple((0, 1) for _ in range(len(avg_return_array))) #no short selling, no leverage
#bounds = tuple((-1, 1) for _ in range(len(avg_return_array))) #short selling, no leverage
bounds = tuple((-100, 100) for _ in range(len(avg_return_array))) #short selling, leverage (still constraind that weights must sum to 1)

#Initial guess
init_guess = [1/len(avg_return_array)] * len(avg_return_array)
#init_guess = -1

# Optimization
optimized_result = minimize(fun=objective,x0=init_guess,args=(covariance_array,avg_return_array),method='SLSQP',bounds=bounds,constraints=constraints)

#See results
optimal_weight=optimized_result.x
optimal_return = np.sum(avg_return_array * optimal_weight)
optimal_variance = np.dot(optimal_weight.T, np.dot(covariance_array, optimal_weight))
optimal_std_dev = np.sqrt(optimal_variance)
optimal_sharpe = -optimized_result.fun
print(f"--- Optimal Portfolio Metrics ---")
print(f"Expected Daily Return: {optimal_return:.6f}")
print(f"Daily Variance:        {optimal_variance:.6f}")
print(f"Daily Volatility:      {optimal_std_dev:.6f}")
print(f"Sharpe Ratio:          {optimal_sharpe:.6f}")
optimal_weight

######### Step 4 - Efficiency frontier
#Define the range of target portfolio return and init the results variable
range_min = avg_return_array.min() * 0.99
range_max = optimal_return
target_returns = np.linspace(range_min, range_max, 100)
target_volatilities = []

#Define a new objective function to minimise the variance
def objective2(weights, covariance):
    return np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))

#Loop that goes through target returns and find the corresponding optimal weights that minimise the variance
for target in target_returns:
    # Add a new constraint: the return MUST equal the current 'target'
    # Note that we use an "ineq" parameter to say that the return must be at least the target
    return_constraint = {'type': 'ineq', 'fun': lambda w, t=target: np.dot(w, avg_return_array) - target}
    eff_constraints = [{'type': 'eq', 'fun': weight_constraint}, return_constraint]
    
    res = minimize(fun=objective2, 
                   x0=init_guess, 
                   args=(covariance_array,), 
                   method='SLSQP', 
                   bounds=bounds, 
                   constraints=eff_constraints)
    
    if res.success:
        target_volatilities.append(res.fun)
    else:
        target_volatilities.append(None) # Handle cases where optimization fails

target_volatilities

#Add some naive allocation for comparison purpose
iterations = 100000
naive_ptf_return = []
naive_ptf_volatilities = []

for i in range(iterations):
    naive_weights = np.random.uniform(low=-1, high=1, size=len(avg_return_array))
    naive_weights[-1] = 1 - np.sum(naive_weights[:-1])
    naive_return=np.sum(avg_return_array * naive_weights)
    naive_vol=np.sqrt(np.dot(naive_weights.T, np.dot(covariance_array, naive_weights)))
    naive_ptf_return.append(naive_return)
    naive_ptf_volatilities.append(naive_vol)

#Plotting
plt.figure(figsize=(10, 6))

# The Cloud
plt.scatter(naive_ptf_volatilities, naive_ptf_return, c='lightgray', alpha=0.3, label='Random Portfolios')

# The Frontier (from your previous step)
plt.plot(target_volatilities, target_returns, color='blue', linewidth=2, label='Efficient Frontier')

# The Optimal Point
plt.scatter(optimal_std_dev, optimal_return, color='red', marker='*', s=200, label='Max Sharpe')

plt.xlabel('Volatility')
plt.ylabel('Return')
plt.legend()
plt.show()