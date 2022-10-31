import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# reading the data
df = pd.read_csv('train.csv')
df.head()
df.info()

# Data Analysis
sns.set_style('darkgrid')
print(df['Survived'].unique())
# sns.countplot(x='Survived', data=df)
# plt.show()

print(df['Pclass'].unique())
# sns.countplot(x='Pclass', data=df)
# plt.show()

# sns.countplot(x='Survived', data=df, hue='Pclass')
# plt.show()

# sns.countplot(x='Sex', data=df)
# plt.show()

# sns.countplot(x='Survived', data=df, hue='Sex')
# plt.show()

# sns.histplot(x='Age', data=df)
# plt.show()

# sns.kdeplot(x='Age', data=df)
# plt.show()

# Data Cleaning
print(df.isnull().sum())
# sns.heatmap(df.isnull())
# plt.show()

print(df['Cabin'].unique())
df.drop('Cabin', axis = 1, inplace=True)
print(df.head())
print(df.isnull().sum())
print(df['Age'].mean())
print(df.groupby('Sex').mean()['Age'])
print(df.groupby('Pclass').mean()['Age'])

# sns.boxplot(x='Pclass', y='Age', data=df)
# plt.show()

def update_age(cols):
    age = cols[0]
    pclass = cols[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 30
        else:
            return 25
    else:
        return age

df['Age'] = df[['Age', 'Pclass']].apply(update_age, axis = 1)
# sns.heatmap(df.isnull())
# plt.show()

df.dropna(inplace=True)
# sns.heatmap(df.isnull())
# plt.show()

sex = pd.get_dummies(df['Sex'], drop_first=True)
print(sex)

embarked = pd.get_dummies(df['Embarked'], drop_first=True)
print(embarked)

print(df.head())

df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace=True)

print(df.head())

df = pd.concat([df, sex, embarked], axis = 1)

print(df.head())

X = df.drop('Survived', axis = 1)
y = df['Survived']

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=1)
print(X_train.shape)

# Evaluating best model
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('NB', GaussianNB()))

results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# LR : 0.811751 max accuracy

# prediction
model = RandomForestClassifier()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
print(accuracy_score(y_test,prediction))
print(prediction)

submission=pd.DataFrame({'Survived':prediction},index=X_test.index)
submission.head()
submission.to_csv('prediction.csv')
