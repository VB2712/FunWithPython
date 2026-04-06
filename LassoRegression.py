##########
# We will work with a random data set specifically created for the purpose of testing multi linear regression with and without lasso technique
# The data is composed of 9 random variables, with the dependent variable calculated as 80% of the indep1 and 20% of indep2 plus some noise
# We expect that the lasso regression should be able to ignore the indep variables that do not influence the dependent variable, with the other approach resulting in overfitting and poor off sample perf

#Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso

#Import and clean the data
df=pd.read_excel('RandomDataLinearRegression.xlsx')
df_cleaned=df[['Dep','Indep1','Indep2','Indep3','Indep4','Indep5','Indep6','Indep7','Indep8','Indep9']]
df_cleaned=df_cleaned.dropna()
df_cleaned.head()
df_cleaned

#Plottings
plt.scatter(df['Dep'],df['Indep1'])
plt.show()

#Split data
X = df_cleaned.drop(['Dep'],axis=1)
Y=df_cleaned['Dep']
X.head()
Y.head()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,shuffle=False)

#Create the linear regression model
LinearRegressor=LinearRegression()
LinearRegressor.fit(X_train,Y_train)
y_pred_test_linear=LinearRegressor.predict(X_test)
y_pred_train_linear=LinearRegressor.predict(X_train)

#Print results linear regression
LinearRegressor.coef_
LinearRegressor.intercept_
r2_score(y_pred_train_linear,Y_train)
r2_score(y_pred_test_linear,Y_test)

#Create the lasso regression model
LassoRegressor=Lasso(alpha=5) #alpha is the lambda. Not that in real life we would run an optimization algo to find the optimal value. Here we know that alpha needs to be very high becasue of how were the dummy data constructed
LassoRegressor.fit(X_train,Y_train)
y_pred_test_lasso=LassoRegressor.predict(X_test)
y_pred_train_lasso=LassoRegressor.predict(X_train)

#Print results lasso regression
LassoRegressor.coef_
LassoRegressor.intercept_
r2_score(y_pred_train_lasso,Y_train)
r2_score(y_pred_test_lasso,Y_test)