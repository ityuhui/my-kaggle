import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
dtrain = pd.read_csv("/kaggle/input/titanic/train.csv")
dtrain

X_train_original = dtrain[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
X_train=pd.get_dummies(X_train_original.dropna(axis=1))
X_train.head()

y_train = dtrain[['Survived']]
y_train.head()

from sklearn.linear_model import LogisticRegression
linreg = LogisticRegression(C=1e5, random_state=0)
linreg.fit(X_train, y_train)

print(linreg.intercept_)
print(linreg.coef_)

dtest = pd.read_csv("/kaggle/input/titanic/test.csv")
X_test_original = dtrain[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
X_test=pd.get_dummies(X_test_original.dropna(axis=1))
X_test.head()

linreg.predict(X_test)
