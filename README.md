==================================================
Overall description
--------------------------------------------------
A collection of Python scripts with all sort of quant finance use cases ranging from basic financial engineering (e.g. curve construction) to machine learning application in trading, to more advanced pricing solutions. Read me lists the different scripts organised between pricing tools and ML models, classified by descending complexity order

==================================================
Pricing Tools
==================================================
--------------------------------------------------
Heston pricer
--------------------------------------------------
Pricing of european vanilla options using Heston pricer via FFT (Fast Fourier Transform), calibrate Heston parameters to provide best fit to european vanilla options, and plot calibration results

--------------------------------------------------
American Option LSM
--------------------------------------------------
Pricer for american style option, but using using LSM model. This is based on Monte Carlo type approach for American options, a recently recent development due to advances in compute power

--------------------------------------------------
Simple Monte Carlo
--------------------------------------------------
Pricing of european vanilla options using MC approach, and test convergence to black-scholes price

--------------------------------------------------
Curve Bootstrap
--------------------------------------------------
Zero curve construction - Quant finance 101

==================================================
AI/ML applications
==================================================

--------------------------------------------------
RNN trading signal
--------------------------------------------------
Generates a trading strategy for EQ stocks using an RNN (LSTM type) model. Uses standard triple barrier method

--------------------------------------------------
FinancialMachineLearning_EQ_Feature
--------------------------------------------------
Inspired by "Advances in Financial Learning" - De Prado. Collects some stock prices from Yahoo Finance, implements data labelling, triple barrier method, features engineering, an XGBOOST model, and some basic back test. Should represent reasonably well how is the data prepared for ML methods in Trading application as of 2026

v2 file: multiple improvements and bug fixes

==================================================
Other - General applications
==================================================

--------------------------------------------------
Portfolio Allocation
--------------------------------------------------
Optimization techniques for an Equity portoflio, seeking optimal risk return profile

--------------------------------------------------
Arima-Garch
--------------------------------------------------
Estimates volatility using Arch/Garch type models

--------------------------------------------------
Lasso Regression
--------------------------------------------------
Basic linear regression with Lasso correction

--------------------------------------------------
Order Book
--------------------------------------------------
A simple script which creates class objects to construct a virtual order book. Not super interesting from a quant finance point of view but more from a basic/fundational Python point of view with the usage of classes


