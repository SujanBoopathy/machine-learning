import pandas as pd

df=pd.read_csv('spam.csv')
df.head()

df.groupby('Category').describe()

df['spam']=df['Category'].apply(lambda x: 1 if x=="spam" else 0)
print(df.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.Message,df.spam)


from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
x_train_count=v.fit_transform(x_train.values)
x_train_count.toarray()[:2]

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train_count,y_train)

x_test_count=v.transform(x_test)
print(model.score(x_test_count,y_test))

emails=['hello world  i am waiting for you', 'good work buddy','Upto 20% discount,exclusive offer just for you. Dont miss this reward']
emails_count=v.transform(emails)
print(model.predict(emails_count))



