# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 22:22:33 2020

@author: Hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

data_initial = pd.read_csv("assistment_challenge_train.csv", header=None, sep='delimiter')
student_id = pd.Series([])
j=1
for i in range(len(data_initial)):
    if math.remainder(i+1,3)==1:
        student_id[i]= j
    elif math.remainder(i+1,3)==2:
        student_id[i]= j 
    elif math.remainder(i+1,3)==0:
        student_id[i]= "res"
        j=j+1
        
data_initial.insert(0, "Student_id", student_id)
data_initial.columns = ["Student_id", "response"]


data1 = data_initial.loc[pd.isnull(data_initial["Student_id"])]
data2 = data_initial.loc[data_initial["Student_id"] == "res"]
data3 = data_initial.loc[data_initial["Student_id"]!="res"]
data3.dropna(subset=["Student_id"], inplace=True)
data3.reset_index()

concept_data = pd.DataFrame(data1.response.str.split(",").tolist(), index=data1.Student_id).stack()
concept_data = concept_data.reset_index([0, "Student_id"])
concept_data.columns = ["Student_id", "concept"]

response_data = pd.DataFrame(data2.response.str.split(",").tolist(), index=data2.Student_id).stack()
response_data = response_data.reset_index([0, "Student_id"])
response_data.columns = ["Student_id", "correct"]

x = np.zeros(shape=(len(concept_data), 110))

for i in range(len(concept_data)):
    p= int(concept_data.iloc[i]["concept"])-1
    x[i][p]=1

correct = response_data.iloc[:, 1:2]
x = np.append(correct, x, axis=1)

k=0
for i in range(0, 1368):
    for j in range(k, k+int(data3.iloc[i]["response"])-1):
        x[j][110] = int(data3.iloc[i]["Student_id"])
    k=j
    
raw_data = pd.DataFrame(x)