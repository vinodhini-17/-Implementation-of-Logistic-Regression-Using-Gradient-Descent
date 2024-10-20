# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
````
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Finally execute the program and display the output.
````
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: vinodhini k
RegisterNumber: 212223230245
 
*/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataset=pd.read_csv("/content/Placement_Data_Full_Class (1).csv")
dataset.head()
```
![image](https://github.com/user-attachments/assets/8b74aebd-ff84-4060-a85a-a5736c6160aa)
```
dataset.tail()
``````
![image](https://github.com/user-attachments/assets/490f4af9-164e-48d6-bcdb-f59bdba68d88)
````
dataset.info()
`````
![image](https://github.com/user-attachments/assets/8a873c83-bdb4-49d1-8f1b-dc0e89033785)
``````
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
`````````
![image](https://github.com/user-attachments/assets/c20b2f1f-fa39-44b2-906d-6f5cd4f3b815)
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
```````
![image](https://github.com/user-attachments/assets/d72c52ac-bdf3-4f20-861f-fd35dd0a4242)
``````
dataset.info()

````````
![image](https://github.com/user-attachments/assets/c8971a57-2bf7-4c44-8814-6a7f66a5999e)

````
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
clf=LogisticRegression()
clf.fit(x_train,y_train)
`````
![image](https://github.com/user-attachments/assets/4a5c0f1a-99c9-4b0d-be55-19298b687ece)
````
y_pred=clf.predict(x_test)
clf.score(x_test,y_test)
````
![image](https://github.com/user-attachments/assets/5d06f4a3-3312-4151-a927-18b63717eca4)
````
from sklearn.metrics import  accuracy_score, confusion_matrix
cf=confusion_matrix(y_test, y_pred)
cf
```
![image](https://github.com/user-attachments/assets/8cfcb49d-a10d-4c18-9a7c-2c39463c8fe2)
````



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming..

