import pandas as pd
import matplotlib.pyplot as plt


train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

# print(train_data.label.astype('category').value_counts)

four=train_data.iloc[3,1:]
print(four.shape)
four=four.values.reshape(28,28)
print(four.shape)
plt.imshow(four,cmap='gray')
plt.show()

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
x=train_data.drop(columns='label')
y=train_data['label']
x=x/255.0
test_data=test_data/255.0
print("train data size:",x.shape)
print("test data size:",test_data.shape)

from sklearn.preprocessing import scale
x_scaled=scale(x)

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.3,train_size=0.2,random_state=10)

linear_model=SVC(kernel='linear')
linear_model.fit(x_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix

y_pred=linear_model.predict(x_test)
print(accuracy_score(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))

nl_model=SVC(kernel='rbf')
nl_model.fit(x_train,y_train)
y_pred=nl_model.predict(x_test)

print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
