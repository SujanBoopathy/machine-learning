import pandas as pd
import numpy as np
import tensorflow
titanic=pd.read_csv('titanic_train.csv')

print(titanic.head())

titanic=titanic.drop(['passenger_id','name','sibsp','parch','ticket','cabin','embarked','boat','body','home.dest'],axis=1)

print(titanic)

titanic=titanic.dropna()
print(titanic)

x=titanic.drop('survived',axis=1)
y=titanic['survived']

x['sex']=x['sex'].map({'male':0,'female':1})


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)

from sklearn import tree
model=tree.DecisionTreeClassifier()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))

print(titanic.drop(['pclass','age','fare'],axis=1).value_counts())

test_data={
    'pclass':[2,3],
    'sex':[0,1],
    'age':[45.0,19.0],
    'fare':[8.665,9.065]
}

test_dataset=pd.DataFrame(test_data,columns=['pclass','sex','age','fare'])
print(model.predict(test_dataset))

def gini_index(survived,total):
    s_prob=survived/total
    ns_prob=1-s_prob
    return (1-((s_prob*s_prob)+(ns_prob*ns_prob)))

print(gini_index(266,675))

print(gini_index(88,429))

print(gini_index(178,246))

