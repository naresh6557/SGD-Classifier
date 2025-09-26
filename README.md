# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.SGD Classifier                                                                                                                                                                                                    
2.Logistic Regression                                                                                                                                                                                                
3.Decision Tree                                                                                                                                                                                                     
4.K-Nearest Neighbors (KNN)                                                                                                                                                                                        

## Program:
1.load the dataset
```
from sklearn.datasets import load_iris
iris_data = load_iris()
```
2.split the data
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris_data.data, iris_data.target, test_size=0.2, random_state=42
)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")
```
3.train the model
```
from sklearn.linear_model import SGDClassifier

# Instantiate an SGDClassifier object with default parameters
sgd_clf = SGDClassifier(random_state=42)

# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)
```
4.evaluate the model
```
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
```



## Output:

1.split the data










<img width="800" height="107" alt="image" src="https://github.com/user-attachments/assets/736cb7bf-36b9-453b-8fbb-2be592dd01be" />






2.train the model











<img width="622" height="107" alt="image" src="https://github.com/user-attachments/assets/076ca35a-56e2-43f4-bc30-c584caf5b54c" />



3.evaluate the model










<img width="1772" height="142" alt="image" src="https://github.com/user-attachments/assets/62ca22a2-e8e5-463b-a65d-420caf49d2eb" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
