#%%
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';') 
#%%
data.quality.value_counts().plot(kind='bar')
#%%
	y = data.quality
x = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=0, 
                                                    stratify=y)

X_train = X_train.drop(['citric acid', 'chlorides'], axis=1)
X_test = X_test.drop([ 'citric acid', 'chlorides'], axis=1)
#%%

pipeline = Pipeline([
                     ('clf', RandomForestClassifier(n_estimators=100, max_depth=25))])

pipeline.fit(X_train, y_train)
print(pipeline.score(X_train, y_train), pipeline.score(X_test, y_test))

score = cross_val_score(pipeline, X_train, y_train, cv=7)
print(score.mean())


# It seems as though this is a very difficult dataset to predict
# Wine quality simply can't be predicted by it's chemical makeup
#with a higher than 70% accuracy