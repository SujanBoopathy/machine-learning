import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def unitStep(v):
    if v>=0:
        return 1
    else:        
        return 0

def perceptronModel(x,w,b):
    y=np.dot(x,w)+b
    return unitStep(y)

def and_logic(x):
    w=np.array([1,1])
    band=-1.5
    return perceptronModel(x,w,band)

def or_logic(x):
    w=np.array([1,1])
    bor=-0.5
    return perceptronModel(x,w,bor)
def not_logic(x):
    w=-1
    b=0.5
    return perceptronModel(x,w,b)
def xor_logic(x):
    y1=and_logic(x)
    y2=or_logic(x)
    y3=not_logic(y1)
    final_x=np.array([y2,y3])
    final_out=and_logic(final_x)
    return final_out

test1=np.array([0,0])
test2=np.array([0,1])
test3=np.array([1,0])
test4=np.array([1,1])

print(xor_logic(test1))
print(xor_logic(test2))
print(xor_logic(test3))
print(xor_logic(test4))
