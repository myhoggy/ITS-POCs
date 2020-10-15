# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:57:45 2020

@author: Hp
"""

import pandas as pd
import math

df = pd.read_csv("Demo1.csv")

rows = len(df)
cols = int((len(df.columns)-1)/6)

for i in range(cols):
    df['L'+str(i+1)] = df['L'+str(i+1)+'Q1']+df['L'+str(i+1)+'Q2']+df['L'+str(i+1)+'Q3']+df['L'+str(i+1)+'Q4']+df['L'+str(i+1)+'Q5']+df['L'+str(i+1)+'Q6']

arr = [0]*rows
df.insert(int(len(df.columns)), "Total Response", arr)
  
for i in range (cols):
    df['Total Response'] += df['L'+str(i+1)]
    
lesson_difficulty = [[0 for i in range(cols)] for j in range(rows)]
correctness_prob = [[0 for i in range(cols)] for j in range(rows)]
shortest_path = [0]*rows
info = []

def sortSecond(val):
    return val[1]

for i in range(rows):
    a = math.log((df['Total Response'][i])/(cols*6)/(1-(df['Total Response'][i])/(cols*6)))
    for j in range(cols):
        lesson_difficulty[i][j] = math.log((1-(df['L'+str(j+1)][i]/6))/(df['L'+str(j+1)][i]/6))
        correctness_prob[i][j] = ((math.e)**(a-lesson_difficulty[i][j]))/(1+(math.e)**(a-lesson_difficulty[i][j]))
    correctness_prob[i] = [item for item in correctness_prob[i] if item<0.75]

for i in range(rows):
    for j in range(len(correctness_prob[i])):
        info.append(('L'+str(j+1), correctness_prob[i][j]*(1-correctness_prob[i][j])))
    info.sort(key=sortSecond)
    shortest_path[i] = info
    info=[]
    
questions_asked = [[[0 for i in range(6)] for j in range(cols)] for k in range(rows)]
questions_available = [[[0 for i in range(3)] for j in range(cols)] for k in range(rows)]

for i in range(rows):
    for j in range(len(shortest_path[i])):
        if shortest_path[i][j][1]<0.05:
            questions_asked[i][j][0] = 'E'
            questions_asked[i][j][1] = 'E'
            questions_asked[i][j][2] = 'E'
            questions_asked[i][j][3] = 'E'
            questions_asked[i][j][4] = 'M'
            questions_asked[i][j][5] = 'H'
            questions_available[i][j][0]+=4
            questions_available[i][j][1]+=1
            questions_available[i][j][2]+=1
        
        elif shortest_path[i][j][1]<0.1:
            questions_asked[i][j][0] = 'E'
            questions_asked[i][j][1] = 'E'
            questions_asked[i][j][2] = 'E'
            questions_asked[i][j][3] = 'M'
            questions_asked[i][j][4] = 'M'
            questions_asked[i][j][5] = 'H'
            questions_available[i][j][0]+=3
            questions_available[i][j][1]+=2
            questions_available[i][j][2]+=1
            
        elif shortest_path[i][j][1]<0.15:
            questions_asked[i][j][0] = 'E'
            questions_asked[i][j][1] = 'E'
            questions_asked[i][j][2] = 'M'
            questions_asked[i][j][3] = 'M'
            questions_asked[i][j][4] = 'H'
            questions_asked[i][j][5] = 'H'
            questions_available[i][j][0]+=2
            questions_available[i][j][1]+=2
            questions_available[i][j][2]+=2
            
        elif shortest_path[i][j][1]<0.2:
            questions_asked[i][j][0] = 'E'
            questions_asked[i][j][1] = 'M'
            questions_asked[i][j][2] = 'M'
            questions_asked[i][j][3] = 'M'
            questions_asked[i][j][4] = 'H'
            questions_asked[i][j][5] = 'H'
            questions_available[i][j][0]+=1
            questions_available[i][j][1]+=3
            questions_available[i][j][2]+=2
            
        else:
            questions_asked[i][j][0] = 'M'
            questions_asked[i][j][1] = 'M'
            questions_asked[i][j][2] = 'M'
            questions_asked[i][j][3] = 'H'
            questions_asked[i][j][4] = 'H'
            questions_asked[i][j][5] = 'H'
            questions_available[i][j][0]+=0
            questions_available[i][j][1]+=3
            questions_available[i][j][2]+=3
            






        