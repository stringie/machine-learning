#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';') 
#%%

y = data.quality
x = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)


X_train = X_train.drop(['fixed acidity', 'citric acid', 'chlorides', 'density'], axis=1)
X_test = X_test.drop(['fixed acidity', 'citric acid', 'chlorides', 'density'], axis=1)
model = RandomForestClassifier(n_estimators=100, max_depth=15)
X_train
#%%

model.fit(X_train, y_train)
print(model.score(X_train, y_train), model.score(X_test, y_test))


# It seems as though this is a very difficult dataset to predict
# Wine quality simply can't be predicted by it's chemical makeup
#with a higher than 70% accuracy