import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

#####Caveats and Comments######
#We implement a simple bootstrap script. The inputs are virtual market instruments, defined with a time bucket and a par yield. We also input the target maturities, that can be richer than the inputs
#We start by doing a linear interpolation of the input instruments to fill the entire target maturities. Then we run a bootstrap algorithm. We finally perform some plotting activity to ensure that the shape of the zero curve and forward rates seem reasonable
#We simplify the implementation by ignoring day count conventions, and we also assume that all inputs and output buckets are on exact year points
#We also ignore economic events such as monetary meetings or turn of year effects
#Kink issue: we observe a spike in the forward rates due to the flat extrapolation of the curve. Industry standards to avoid is to not apply the interpolation to the par rates but instead either to use a cubic splin or more advanced interpolation techniques, otherwise to use the log of DF to produce a step like forward rates

#####Inputs#####
#Market instruments: maturities and par rates
market_maturities = np.array([1, 2, 3, 5])
market_rates = np.array([0.025, 0.03, 0.035, 0.05])
#Target maturities
target_maturities=np.array([1,2,3,4,5,6,7,8,9,10,20])

#####Step 1 - Interpolate the market rates #####
#longest maturity we are solving for - will be used in the step 2
max_T=np.max(target_maturities)
all_years = np.arange(1, max_T + 1)
#Interpolated market rates - using linear interpolation
interpolated_market_rates=np.interp(all_years,market_maturities,market_rates)

#####Step 2 - Bootstrapping#####
#Initiate the target variables
solved_zero_rates=[]
solved_df=[]
one_year_forward=[]
#Initiate loop. Note that the statement 'for i, T in enumerate' is not a double loop; it is a single loop but that gives the index i and the value T so we dont need to write T=all_years[i] in the loop
for i, T in enumerate(all_years):

    coupon=interpolated_market_rates[i]

    def objective_function(r):
        #Sum of discount factors for all maturities until current maturity
        sdf=sum(solved_df)
        #Discount factor for the maturity that we are solving for
        df=1/(1+r)**T
        #Par swap NPV must be 1 we are solving for 1=c*sdf+c*DF(T)+DF(T) <=> 0=cx(sdf+DF(T))+DF(T)-1
        return (coupon*(sdf+df)+(df))-1
    
    #Solve for the current rate (looking between -10% and +40%)
    zero_rate=brentq(objective_function,-0.1,0.4)

    solved_zero_rates.append(zero_rate)
    solved_df.append(1/(1+zero_rate)**T)

    # Calculate the time step from the previous maturity
    if i == 0:
        # For the first period, the forward is just the spot zero rate
        fwd = solved_zero_rates[0]
    else:
        delta_t = all_years[i] - all_years[i-1]
        fwd = (solved_df[i-1] / solved_df[i] - 1) / delta_t

    # Store it to plot later
    one_year_forward.append(fwd)
    
#####Step 3 - Plotting#####
plt.figure(figsize=(10, 5))
plt.plot(all_years, interpolated_market_rates, 'o--', label='Par Swap Curve (Input)', alpha=0.7)
plt.plot(all_years, solved_zero_rates, 'r-', label='Zero Rate Curve (Bootstrapped)')
plt.plot(all_years, one_year_forward, 'g-', label='One year forward rate')
plt.xlabel('Maturity (Years)')
plt.ylabel('Rate')
plt.title('Yield Curve Bootstrapping')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('yield_curve.png')
plt.show()