# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Finally execute the program and display the output.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: vinodhini k
RegisterNumber:  212223230245
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataset=pd.read_csv("/content/Placement_Data_Full_Class (1).csv")
dataset.head()
```
![image](https://github.com/user-attachments/assets/c73780bb-7e55-4da2-82c4-2724242adcf5)
````
dataset.tail()
```
![image](https://github.com/user-attachments/assets/7930feb8-ab02-480a-8e65-d73db5d38724)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/f7352064-2c0a-4ce9-bd40-e953f08c8d31)
````
dataset=dataset.drop('sl_no',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
````
![image](https://github.com/user-attachments/assets/2db36efc-e3c0-4f0c-b701-901f37842077)
````
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
````
![image](https://github.com/user-attachments/assets/2cc0f5d4-628d-49c2-a19e-20fb94a16bff)
````
dataset.info()
```
![image](https://github.com/user-attachments/assets/22d23254-59cb-487a-b2e1-f043a16b151c)
````
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
clf=LogisticRegression()
clf.fit(x_train,y_train)
````
![image](https://github.com/user-attachments/assets/d2e4bcd1-c11a-4695-8167-875748990fe9)
````
y_pred=clf.predict(x_test)
clf.score(x_test,y_test)
````
![image](https://github.com/user-attachments/assets/b447d792-fcff-47c5-9a8f-d79cc9c04467)
````
from sklearn.metrics import  accuracy_score, confusion_matrix
cf=confusion_matrix(y_test, y_pred)
cf
````
![image](https://github.com/user-attachments/assets/5b3861b3-edc9-4e17-bf11-de211aa6de96)
````
accuracy=accuracy_score(y_test, y_pred)
accuracy
`````
![image](https://github.com/user-attachments/assets/e2f21932-8775-49d1-a010-c4ea5a57b13b)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

