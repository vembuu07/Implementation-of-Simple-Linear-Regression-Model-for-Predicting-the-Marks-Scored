# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas
   

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: vembarasan.p
RegisterNumber:  212223220123
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
## Dataset
![dataset](https://github.com/vembuu07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150772461/35548586-783c-4c43-904b-c38a2bb47025)
## Head values
![head](https://github.com/vembuu07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150772461/0b1cad91-e041-43e3-b542-c3e27ebdd535)
## Tail values
![tail](https://github.com/vembuu07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150772461/1cb9ce7f-1dc6-4c5a-affb-208aaf02d046)
## X and Y values
![xyvalues](https://github.com/vembuu07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150772461/2d0000c4-1a25-43aa-8219-85bb3f81bc28)
## Predication values of X and Y
![predict ](https://github.com/vembuu07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150772461/6359e040-3b8c-4d21-bc81-9d5121729133)
## MSE,MAE and RMSE
![values](https://github.com/vembuu07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150772461/9e950a86-cb46-4a34-8b21-6fd553458322)
## Training set
![train](https://github.com/vembuu07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150772461/13e10aac-705c-4932-baf4-0a7ca127d3e5)
## Training set
![test](https://github.com/vembuu07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150772461/ec19bddc-4068-41df-be17-9de2cf4ea92e)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
