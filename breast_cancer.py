from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

breast_cancer=pd.read_csv('wisc_bc_data.csv')

print(breast_cancer.head())

del breast_cancer['id']

print(breast_cancer.head())

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(breast_cancer.loc[:,breast_cancer.columns!='diagnosis'],breast_cancer['diagnosis'],stratify=breast_cancer['diagnosis'],random_state=66)

boosting=GradientBoostingClassifier()
boosting.fit(x_train,y_train)

max_boosting=GradientBoostingClassifier(max_depth=1)
max_boosting.fit(x_train,y_train)

print(f"accuracy:  {format(max_boosting.score(x_test,y_test),'.4f')} ")

breast_cancer_features=[x for i,x in enumerate(breast_cancer.columns) if i!=30 ]

def graph_plot(model):
    n_features=30
    plt.figure(figsize=(10,5))
    plt.barh(range(n_features),model.feature_importances_,align='center',color=['black'])
    plt.yticks(np.arange(n_features),breast_cancer_features)
    plt.title("feature imp graph")
    plt.xlabel('feature imp')
    plt.ylabel('feature')
    plt.ylim(-1,n_features)

graph_plot(max_boosting)
plt.show()