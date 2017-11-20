#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

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
#%%
X_train
#%%

pipeline = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(random_state=1))])

# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# param_grid = [{'clf__C': param_range, 
#                'clf__kernel': ['linear']},
#              {'clf__C': param_range, 
#                'clf__gamma': param_range, 
#                'clf__kernel': ['rbf']}]

# gs = GridSearchCV(estimator=pipeline, 
#                             param_grid=param_grid, 
#                             scoring='accuracy', 
#                             cv=4)
# scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=2)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# pipeline.fit(X_train, y_train)
# print(pipeline.score(X_train, y_train), pipeline.score(X_test, y_test))

score = cross_val_score(pipeline, X_train, y_train, cv=7)
print(score.mean())

# It seems as though this is a very difficult dataset to predict
# Wine quality simply can't be predicted by it's chemical makeup
#with a higher than 70% accuracy