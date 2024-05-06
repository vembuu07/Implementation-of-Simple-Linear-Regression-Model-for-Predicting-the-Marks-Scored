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

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VEMBARASAN P
RegisterNumber:  212223220123
*/
```
```
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
## dataset
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/3601ca3e-1300-49a7-8107-37fd8685caa8)

## head and tail values
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/22676f1c-6f6c-431a-887a-c4ceb46e508a)

## X and Y values
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/18ee5a39-6710-4ab3-8a4b-3873be89221d)

## Predication values of X and Y
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/0a17a630-8872-40bf-b77c-04b5ebdd2b13)

## MSE,MAE and RMSE
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/84ff9855-0058-4cb6-935a-f9909217f785)

## Training Set
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/e3fe1065-e497-440d-8fc2-d86e8d98181c)

## Test Set Graph
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/5d9f0ced-bd0f-4c10-88d6-4fbeb3938db7)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
