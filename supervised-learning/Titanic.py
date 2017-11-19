#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def dropper(set, *args):
    for arg in args:
        set = set.drop(arg, axis=1)
    return set

train = pd.read_csv('/home/string/dev/machine-learning/supervised-learning/train.csv', index_col='PassengerId')
test = pd.read_csv('/home/string/dev/machine-learning/supervised-learning/test.csv', index_col='PassengerId')

train = dropper(train, 'Cabin', 'Fare', 'Ticket')
test = dropper(test, 'Cabin', 'Fare', 'Ticket')

train['Sex'] = train['Sex'].apply(lambda x: 0 if x == 'male' else 1)
train['Age'] = train.Age.fillna(train.Age.mean())
train['Embarked'] = train.Embarked.fillna('S')

train['Title'] = train.Name.str.extract('([A-Za-z]+)\.', expand=False)
train.loc[train.Title == 'Mlle', 'Title'] = 'Miss'
train.loc[train.Title == 'Mme', 'Title'] = 'Mrs'
train.loc[train.Title == 'Ms', 'Title'] = 'Miss'
rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Don', 'Jonkheer', 'Capt', 'Lady', 'Sir']
train.Title = train.Title.replace(rare_titles, 'Rare')

age_by_title = train.groupby('Title').Age.mean()

train.loc[train.Age.isnull() & (train.Title == 'Mr'), 'Age'] = age_by_title['Mr']
train.loc[train.Age.isnull() & (train.Title == 'Mrs'), 'Age'] = age_by_title['Mrs']
train.loc[train.Age.isnull() & (train.Title == 'Miss'), 'Age'] = age_by_title['Miss']
train.loc[train.Age.isnull() & (train.Title == 'Master'), 'Age'] = age_by_title['Master']
train.loc[train.Age.isnull() & (train.Title == 'Rare'), 'Age'] = age_by_title['Rare']

train['FamiliSize'] = train.Parch + train.SibSp + 1
train = dropper(train, 'Parch', 'SibSp')
train

#%%
X = dropper(train, 'Name', 'Embarked', 'Survived', 'Title')
y = train['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


model = RandomForestClassifier(n_estimators=30, max_depth=4)

# search = GridSearchCV(model, {'n_estimators': [10,20, 30,40, 50,60, 70,80,90, 100,110, 130],
#                                 'max_depth': [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]})


model.fit(x_train, y_train)
print(model.score(x_train, y_train), model.score(x_test, y_test))
# pd.DataFrame(search.cv_results_)[['rank_test_score', 'mean_test_score', 'params']].sort_values('rank_test_score').head(10)
