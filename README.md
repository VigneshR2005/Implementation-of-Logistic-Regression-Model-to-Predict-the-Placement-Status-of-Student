# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Find the null and duplicate values.
3. Using logistic regression find the predicted values of accuracy , confusion matrices.
4. Display the results.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: R Vignesh
RegisterNumber:212222230172
**
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```
## Output:
## 1.Placement Data
![9](https://github.com/Senthamil1412/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119120228/5a6b3c5c-3935-408d-9538-117f258255d4)
## 2.Salary Data
![99](https://github.com/Senthamil1412/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119120228/69129d48-5f6d-426e-9548-aac730036b4b)
## 3.Checking the null function()
![999](https://github.com/Senthamil1412/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119120228/8f5964fa-3c0c-4ae4-a2cc-521f69efc831)
## 4.Data Duplicate

![9999](https://github.com/Senthamil1412/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119120228/fd7196cf-812e-49ee-a819-44dc10c9669f)
## 5.Print Data

![99999](https://github.com/Senthamil1412/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119120228/0dec6480-a93b-42e2-a4c4-73d5f2b04d28)
## 6.Data Status

![999999](https://github.com/Senthamil1412/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119120228/d50d39ac-6a3c-47c4-982b-bfe7e15c8f98)
## 7.y_prediction array

![9999999](https://github.com/Senthamil1412/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119120228/9b61513f-972b-4ee0-ad60-1121ef24ce5a)
## 8.Accuracy value

![2](https://github.com/Senthamil1412/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119120228/8cf872ee-4da8-4749-b242-f805aec7e3a0)
## 9.Confusion matrix

![22](https://github.com/Senthamil1412/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119120228/f89fa957-80f9-4fd9-a72c-813e51005f52)

## 10.Classification Report
![222](https://github.com/Senthamil1412/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119120228/96eaabe3-0b2f-4455-8dc8-925fabd69454)

## 11.Prediction of LR
![33](https://github.com/Senthamil1412/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119120228/a8c71717-3a08-4c0d-826b-95b0c4d1dbcc)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
