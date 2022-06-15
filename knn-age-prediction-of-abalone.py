#Exp-3 : KNN algorithm to find the age of the abalone

import pandas as pd
import numpy as np

url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/abalone/abalone.data"
)
abalone=pd.read_csv(url,header=None)

abalone.head()

abalone.columns=[
        'Sex',
        'Length',
        'Diameter',
        'Height',
        'Whole weight',
        'Shucked weight',
        'Viscera weight',
        'Shell weight',
        'Rings',   
]

abalone=abalone.drop('Sex',axis=1)

correlation_matrix=abalone.corr()
correlation_matrix['Rings']

X=abalone.drop('Rings',axis=1)
X=X.values
Y=abalone['Rings']
Y=Y.values

new_data_point=np.array([
            0.5402,
            0.2654,
            0.1022,
            0.0898,
            1.8933,
            0.3423,
            0.8783
])

distances=np.linalg.norm(X-new_data_point,axis=1)

k=11
k_neighbour_ids=distances.argsort()[:k]
k_neighbour_ids

k_nearest_rings=Y[k_neighbour_ids]
k_nearest_rings

import scipy.stats 
print(scipy.stats.mode(k_nearest_rings))

# prediction=k_nearest_rings.mean()
# print(prediction*1.5)

print("The age of the new abalone : ",k_nearest_rings[8])