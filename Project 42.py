import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
diabetes_dataset = pd.read_csv('diabetes_prediction_dataset.csv')

print(diabetes_dataset)
diabetes_dataset.head(10)
columns_to_drop = ['smoking_history','gender']
diabetes_dataset.drop(columns=columns_to_drop, inplace=True)
diabetes_dataset.shape
diabetes_dataset.isnull().sum()
diabetes_dataset.describe()
diabetes_dataset['diabetes'].value_counts()
diabetes_dataset.groupby('diabetes').mean()
X=diabetes_dataset.drop(columns='diabetes',axis=1)
Y=diabetes_dataset['diabetes']
print(X)
sc= StandardScaler()
sc.fit(X)
X_train , X_test,Y_train,Y_test = train_test_split(X,Y , test_size=0.2, stratify=Y , random_state=0)
print(X.shape,X_train.shape,X_test.shape)
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, Y_train)

# Make predictions on the test data
y_pred = logistic_reg.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)
new_data_diabetic = [[65, 1, 1, 30.5, 7.2, 250]]
new_data_diabetic = sc.transform(new_data_diabetic)

# Use the trained model to make a prediction
prediction = logistic_reg.predict(new_data_diabetic)
if prediction[0] == 1:
    print('THE RESULT IS DIABETIC')
else:
    print('THE RESULT IS NOT DIABETIC')