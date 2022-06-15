import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split


df = pd.read_csv("titanic_train.csv")
df.drop(['passenger_id','name','sibsp','parch','ticket','cabin','embarked','boat','body','home.dest'],axis='columns',inplace=True)
inputs = df.drop('survived',axis='columns')
target = df.survived
inputs.sex=inputs.sex.map({'male':1,'female':2 })
inputs.age=inputs.age.fillna(inputs.age.mean())
inputs.fare=inputs.fare.fillna(inputs.fare.mean())
print(inputs.head())
print('-'*20)
print(target.head())
X_train, X_test, y_train, y_test = train_test_split(inputs, target,test_size=0.2)
model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
test_data = {'pclass': [3,3],  
                'sex': [2,1],
                'age':[29.519847,38.000000],
                'fare':[7.7333,8.6625]}
check = pd.DataFrame(test_data, columns = ['pclass', 'sex', 'age','fare']) 
res = model.predict(check)
check["survived"] = res
print('-'*20)
print("Predict Results:") 
print(check)
def get_gini_impurity(survived_count, total_count):
    s_prob = survived_count/total_count
    ns_prob = (1 - s_prob)
    return (1-((s_prob*s_prob)+(ns_prob*ns_prob)))
print("Overall Gini Impurity")
print(get_gini_impurity(342, 891))
print("Gini Impurity of Men")
print(get_gini_impurity(109, 577))
print("Gini Impurity of Women")
print( get_gini_impurity(233, 314))
# from IPython.display import Image as Image
# from six import StringIO
# from sklearn.tree import export_graphviz
# import pydotplus
# dot_data=StringIO()
# export_graphviz(model,out_file=dot_data,filled=True,rounded=True,special_characters=True,
#               class_names=['0','1'])
# graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('tree.png')
# Image(graph.create_png())