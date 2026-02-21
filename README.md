# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program and import required libraries.
2.Create and load the employee dataset into a DataFrame.
3.Separate input features and target variable (Churn).
4.Split the dataset into training and testing sets.
5.Train the Decision Tree Classifier model and predict test results.
6.Evaluate accuracy, display classification report, plot the tree, and stop.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Yogesh G
RegisterNumber:  25009804
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

data = {
    "Age": [22, 35, 45, 28, 50, 41, 30, 26, 48, 33],
    "Salary": [20000, 50000, 70000, 30000, 90000, 65000, 40000, 25000, 85000, 48000],
    "YearsAtCompany": [1, 7, 15, 2, 20, 10, 4, 1, 18, 6],
    "JobSatisfaction": [2, 4, 3, 2, 5, 4, 3, 1, 5, 3],
    "Churn": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No"]
}

df = pd.DataFrame(data)
X = df[["Age", "Salary", "YearsAtCompany", "JobSatisfaction"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=["Age", "Salary", "YearsAtCompany", "JobSatisfaction"],
    class_names=["No", "Yes"],
    filled=True
)
plt.title("Decision Tree Classifier for Employee Churn Prediction")
plt.show()
```

## Output:
<img width="523" height="202" alt="Screenshot 2026-02-21 111344" src="https://github.com/user-attachments/assets/d8d79115-ebc3-4008-821d-d2c3133a7d65" />
<img width="945" height="595" alt="Screenshot 2026-02-21 111353" src="https://github.com/user-attachments/assets/002b2366-2d0c-4f7e-9830-f9d79cf16fed" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
