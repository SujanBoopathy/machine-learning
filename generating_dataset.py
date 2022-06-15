import pandas as p 

df=p.DataFrame({
    'CGPA' : [8,6,9,7],
    'Attendance' : [89,90,76,56],
    'Income' : [30000,50000,600000,800000],
    'HSC' :[90,89,67,64]
})

print(df.head())
print()
def dataset(a,b,c,d):
    score=0
    if(a>7):
        score+=1
    if(b>80):
        score+=1
    if(c>50000):
        score+=1
    if(d>=65):
        score+=1
    return score

df['scholar_ship_score']=df.apply(lambda x:dataset(x['CGPA'],x['Attendance'],x['Income'],x['HSC']),axis=1)
print(df.head())
