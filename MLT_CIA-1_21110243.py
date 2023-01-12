#INFERENCE IS GIVEN AT THE END OF THE PYTHON FILE!!!
import numpy as np
import matplotlib.pyplot as plt
import random
#%%
#Generate Random points

#x = np.random.randint(1000, size = 1000)
#y = np.random.randint(1000, size = 1000)

l = 1000
mu = 0
sigma = 0.8
x = np.random.normal(mu, sigma, l) 
y = x + 2 * np.random.normal(mu, sigma, l) 

#print(len(x))
#print(len(y))

#%%
#Linear Regression
xm = x.mean()
ym = y.mean()

temp1 = 0
temp2 = 0


x_train = x[:700]
x_test =  x[700:]
y_train = y[:700]
y_test =  y[700:]

#print(len(x_test))

for i in range(700):
    temp1 += (x[i] - xm)*(y[i] - ym)
    temp2 += (x[i] - xm)**2

b1 = temp1/temp2
b0 = ym - (b1*xm)
y_pred = np.zeros(300)

for i in range(300):
    y_pred[i] = b0 + (b1*x_test[i])
print("Linear Regression Model:")
print("m: ",b1,"c: ", b0)



plt.scatter(x,y)
plt.plot(x_test, y_pred, c = 'r')
plt.show()

cost = 0
for i in range(300):
    cost += (y_test[i] - y_pred[i])**2
cost /= l
print("Mean Squared Error: ",cost)
print()

#%%
#Gradient Descent
Dm = 0
Dc = 0
threshold = 0.001
L = 0.01
m = random.random()
c = random.random()
#print(m)
#print(c)
temp = 10

for i in range(500):
    temp = 0
    for i in range(l):
        Dm += x[i] *(y[i] - ((m*x[i]) + c))
        Dc += (y[i] - ((m*x[i]) + c))
        temp += ((y[i] - (m*x[i] + c))**2)
    Dm = -2/l * (Dm)
    Dc = -2/l * (Dc)
    temp/=l
    m = m - (L*Dm)
    c = c - (L*Dc)
print("Linear Regression Model with Gradient Descent:")
print("m: ",m,"c: ",c)
print("Mean Squared Error: ", temp)
print()

#%%
#Sklearn Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=0)
reg = LinearRegression()
reg.fit(x_train.reshape(-1,1),y_train)
y_pred = reg.predict(x_test.reshape(-1,1))

print("Linear Regression Model with sklearn:")
print("m: ",reg.coef_[0], "c: ",reg.intercept_)

from sklearn.metrics import r2_score, mean_squared_error

#print("r2 score: ",r2_score(y_test,y_pred))

import math
print("Mean Squared Error: ",math.sqrt(mean_squared_error(y_test,y_pred)))
print()

#%%
"""
Inference:
    Linear Regression is based on simple formulation and implementation, it doesn't depend on any parameters
    However, Gradient Descent depends on the learning rate and the number of iterations
    For eg, 
    For 100 iterations, the values of m and c are much worse than for 500 iterations
    Similarly, learning rate shouldn't be too high or low, ideally for this case it's 0.01
"""
    





    
