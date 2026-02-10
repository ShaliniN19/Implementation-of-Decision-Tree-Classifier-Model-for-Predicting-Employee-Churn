# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by:shalini n
RegisterNumber:  212224040305
*/
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("/content/Employee.csv")
```
```
data.head()
```
<img width="1320" height="312" alt="image" src="https://github.com/user-attachments/assets/36d148e3-032d-4107-a328-265ca7a3bb33" />
```
data.info()
```
<img width="611" height="408" alt="image" src="https://github.com/user-attachments/assets/5455ac0d-2310-4381-a61e-2d834fac651e" />
```
data.isnull().sum()
```
<img width="446" height="564" alt="image" src="https://github.com/user-attachments/assets/99ebc98d-8b27-4c8c-9e43-6234bc8837f7" />
```
data["left"].value_counts()
```
<img width="285" height="254" alt="image" src="https://github.com/user-attachments/assets/51415129-5849-401d-8cd0-5c7f3009b8f3" />
```from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
```
data.head()
```
<img width="1346" height="358" alt="image" src="https://github.com/user-attachments/assets/e33591e0-1ace-46e4-ae19-b73f7717adb1" />
```
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
```
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
<img width="306" height="37" alt="image" src="https://github.com/user-attachments/assets/dac6b6e9-ddae-4f0d-a79f-945ed408a07c" />
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(15,10))
plot_tree(dt,feature_names=x.columns,class_names=['stayed','left'],filled=True)
```
<img width="1382" height="827" alt="image" src="https://github.com/user-attachments/assets/94735c9e-c7c1-414d-8614-920e7f0694f8" />









## Output:
![decision tree classifier model](sam.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
