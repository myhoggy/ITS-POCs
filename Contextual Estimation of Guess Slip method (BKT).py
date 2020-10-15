# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 22:18:07 2020

@author: Hp
"""

import math as m
import numpy as np
import pandas as pd

raw_data = pd.read_csv('./data.csv', sep=',')
raw_data.shape

raw_data.head(3)

mod1_data = raw_data.set_index(['Student','StepID'])

mod1_data.sort_index(inplace=True)

mod1_data.insert(2,'P(L)1',0.0)
mod1_data.insert(3,'P(C)1',0.0)
mod1_data.insert(4,'P(S)1',0.0)
mod1_data.insert(5,'P(G)1',0.0)

mod1_data.insert(7,'P(L)2',0.0)
mod1_data.insert(8,'P(C)2',0.0)
mod1_data.insert(9,'P(S)2',0.0)
mod1_data.insert(10,'P(G)2',0.0)


mod1_data.insert(12,'P(L)3',0.0)
mod1_data.insert(13,'P(C)3',0.0)
mod1_data.insert(14,'P(S)3',0.0)
mod1_data.insert(15,'P(G)3',0.0)


mod1_data.insert(17,'P(L)4',0.0)
mod1_data.insert(18,'P(C)4',0.0)
mod1_data.insert(19,'P(S)4',0.0)
mod1_data.insert(20,'P(G)4',0.0)


mod1_data.insert(22,'P(L)5',0.0)
mod1_data.insert(23,'P(C)5',0.0)
mod1_data.insert(24,'P(S)5',0.0)
mod1_data.insert(25,'P(G)5',0.0)


mod1_data.insert(27,'P(L)6',0.0)
mod1_data.insert(28,'P(C)6',0.0)
mod1_data.insert(29,'P(S)6',0.0)
mod1_data.insert(30,'P(G)6',0.0)


mod1_data.insert(32,'P(L)7',0.0)
mod1_data.insert(33,'P(C)7',0.0)
mod1_data.insert(34,'P(S)7',0.0)
mod1_data.insert(35,'P(G)7',0.0)

mod1_data.insert(37,'P(L)8',0.0)
mod1_data.insert(38,'P(C)8',0.0)
mod1_data.insert(39,'P(S)8',0.0)
mod1_data.insert(40,'P(G)8',0.0)

mod1_data.insert(42,'P(L)9',0.0)
mod1_data.insert(43,'P(C)9',0.0)
mod1_data.insert(44,'P(S)9',0.0)
mod1_data.insert(45,'P(G)9',0.0)

mod1_data.insert(47,'P(L)10',0.0)
mod1_data.insert(48,'P(C)10',0.0)
mod1_data.insert(49,'P(S)10',0.0)
mod1_data.insert(50,'P(G)10',0.0)

P_L0 = 0.5
P_T = 0.1
P_S0 = 0.1
P_G0 = 0.1

def P_L_func ( correct, P_L_previous, S, G):
  
  if correct==1:
    P_L_obs = (P_L_previous*(1-S))/(P_L_previous*(1-S) + (1-P_L_previous)*(1-G))
  else:
    P_L_obs = (P_L_previous*S)/(P_L_previous*S + (1-P_L_previous)*(1-G))
  
  P_L_current = P_L_obs + (1-P_L_obs)*P_T
  
  return P_L_current

def P_C_func (P_L_previous, S, G):
  P_C_current = P_L_previous*(1-S) + (1-P_L_previous)*G
  
  return P_C_current

def P_S_func (P_L_previous, S, G, A1, A2):
    if A1==1 and A2==1:
        a = (1-S)*(1-S)
        b = P_T*(1-S)*(1-S) + (1-P_T)*P_T*G*(1-S) + (1-P_T)*(1-P_T)*G*G
    elif A1==1 and A2==0:
        a = G*(1-S)
        b = P_T*(1-S)*S + (1-P_T)*P_T*G*S + (1-P_T)*(1-P_T)*G*(1-G)
    elif A1==0 and A2==1:
        a = G*(1-S)
        b = P_T*S*(1-S) + (1-P_T)*P_T*(1-G)*(1-S) + (1-P_T)*(1-P_T)*(1-G)*G
    else:
        a = G*G
        b = P_T*S*S + (1-P_T)*P_T*(1-G)*S + (1-P_T)*(1-P_T)*(1-G)*(1-G)
    
    c = P_L_previous*a + (1-P_L_previous)*b
    
    return (a*P_L_previous)/c

def P_G_func (P_L_previous, S, G, A1, A2):
    if A1==1 and A2==1:
        a = (1-S)*(1-S)
        b = P_T*(1-S)*(1-S) + (1-P_T)*P_T*G*(1-S) + (1-P_T)*(1-P_T)*G*G
    elif A1==1 and A2==0:
        a = G*(1-S)
        b = P_T*(1-S)*S + (1-P_T)*P_T*G*S + (1-P_T)*(1-P_T)*G*(1-G)
    elif A1==0 and A2==1:
        a = G*(1-S)
        b = P_T*S*(1-S) + (1-P_T)*P_T*(1-G)*(1-S) + (1-P_T)*(1-P_T)*(1-G)*G
    else:
        a = G*G
        b = P_T*S*S + (1-P_T)*P_T*(1-G)*S + (1-P_T)*(1-P_T)*(1-G)*(1-G)
    
    c = P_L_previous*a + (1-P_L_previous)*b
    
    return 1-((a*P_L_previous)/c)

for Student, stuInfo in mod1_data.groupby(level=[0]):
  rows = len(stuInfo.index)
  row_loc = 0

  print('Student ID: %s' %(Student))
  

  stuInfo.iloc[0,stuInfo.columns.get_loc('P(L)1')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(C)1')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(S)1')] = P_S0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(G)1')] = P_G0

  stuInfo.iloc[0,stuInfo.columns.get_loc('P(L)2')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(C)2')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(S)2')] = P_S0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(G)2')] = P_G0

  stuInfo.iloc[0,stuInfo.columns.get_loc('P(L)3')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(C)3')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(S)3')] = P_S0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(G)3')] = P_G0

  stuInfo.iloc[0,stuInfo.columns.get_loc('P(L)4')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(C)4')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(S)4')] = P_S0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(G)4')] = P_G0

  stuInfo.iloc[0,stuInfo.columns.get_loc('P(L)5')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(C)5')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(S)5')] = P_S0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(G)5')] = P_G0

  stuInfo.iloc[0,stuInfo.columns.get_loc('P(L)6')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(C)6')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(S)6')] = P_S0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(G)6')] = P_G0

  stuInfo.iloc[0,stuInfo.columns.get_loc('P(L)7')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(C)7')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(S)7')] = P_S0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(G)7')] = P_G0
  
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(L)8')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(C)8')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(S)8')] = P_S0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(G)8')] = P_G0

  stuInfo.iloc[0,stuInfo.columns.get_loc('P(L)9')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(C)9')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(S)9')] = P_S0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(G)9')] = P_G0

  stuInfo.iloc[0,stuInfo.columns.get_loc('P(L)10')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(C)10')] = P_L0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(S)10')] = P_S0
  stuInfo.iloc[0,stuInfo.columns.get_loc('P(G)10')] = P_G0
  

  for index, row in stuInfo.iterrows():
    

    if row_loc >0:
    
      
      if stuInfo.iloc[row_loc]['L1'] == 0.0:
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)1')] = stuInfo.iloc[row_loc-1]['P(L)1']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)1')] = stuInfo.iloc[row_loc-1]['P(C)1']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)1')] = stuInfo.iloc[row_loc-1]['P(S)1']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)1')] = stuInfo.iloc[row_loc-1]['P(G)1']
      else:
        if stuInfo.iloc[row_loc]['Correct']==0.0 and (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)1')] = P_S_func(stuInfo.iloc[row_loc-1]['P(L)1'], stuInfo.iloc[row_loc-1]['P(S)1'], stuInfo.iloc[row_loc-1]['P(G)1'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)1')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)1')]
        elif (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)1')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)1')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)1')] = P_G_func(stuInfo.iloc[row_loc-1]['P(L)1'], stuInfo.iloc[row_loc-1]['P(S)1'], stuInfo.iloc[row_loc-1]['P(G)1'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
        else:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)1')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)1')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)1')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)1')]
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)1')] = P_L_func ( stuInfo.iloc[row_loc]['Correct'], stuInfo.iloc[row_loc-1]['P(L)1'], stuInfo.iloc[row_loc-1]['P(S)1'], stuInfo.iloc[row_loc-1]['P(G)1'])
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)1')] = P_C_func ( stuInfo.iloc[row_loc-1]['P(L)1'], stuInfo.iloc[row_loc-1]['P(S)1'], stuInfo.iloc[row_loc-1]['P(G)1'])
    
    
     
      if stuInfo.iloc[row_loc]['L2'] == 0.0:
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)2')] = stuInfo.iloc[row_loc-1]['P(L)2']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)2')] = stuInfo.iloc[row_loc-1]['P(C)2']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)2')] = stuInfo.iloc[row_loc-1]['P(S)2']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)2')] = stuInfo.iloc[row_loc-1]['P(G)2']
      else:
        if stuInfo.iloc[row_loc]['Correct']==0.0 and (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)2')] = P_S_func(stuInfo.iloc[row_loc-1]['P(L)2'], stuInfo.iloc[row_loc-1]['P(S)2'], stuInfo.iloc[row_loc-1]['P(G)2'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)2')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)2')]
        elif (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)2')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)2')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)2')] = P_G_func(stuInfo.iloc[row_loc-1]['P(L)2'], stuInfo.iloc[row_loc-1]['P(S)2'], stuInfo.iloc[row_loc-1]['P(G)2'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
        else:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)2')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)2')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)2')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)2')]
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)2')] = P_L_func ( stuInfo.iloc[row_loc]['Correct'], stuInfo.iloc[row_loc-1]['P(L)2'], stuInfo.iloc[row_loc-1]['P(S)2'], stuInfo.iloc[row_loc-1]['P(G)2'])
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)2')] = P_C_func ( stuInfo.iloc[row_loc-1]['P(L)2'], stuInfo.iloc[row_loc-1]['P(S)2'], stuInfo.iloc[row_loc-1]['P(G)2'])      
    
    
      
      if stuInfo.iloc[row_loc]['L3'] == 0.0:
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)3')] = stuInfo.iloc[row_loc-1]['P(L)3']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)3')] = stuInfo.iloc[row_loc-1]['P(C)3']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)3')] = stuInfo.iloc[row_loc-1]['P(S)3']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)3')] = stuInfo.iloc[row_loc-1]['P(G)3']
      else:
        if stuInfo.iloc[row_loc]['Correct']==0.0 and (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)3')] = P_S_func(stuInfo.iloc[row_loc-1]['P(L)3'], stuInfo.iloc[row_loc-1]['P(S)3'], stuInfo.iloc[row_loc-1]['P(G)3'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)3')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)3')]
        elif (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)3')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)3')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)3')] = P_G_func(stuInfo.iloc[row_loc-1]['P(L)3'], stuInfo.iloc[row_loc-1]['P(S)3'], stuInfo.iloc[row_loc-1]['P(G)3'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
        else:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)3')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)3')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)3')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)3')]
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)3')] = P_L_func ( stuInfo.iloc[row_loc]['Correct'], stuInfo.iloc[row_loc-1]['P(L)3'], stuInfo.iloc[row_loc-1]['P(S)3'], stuInfo.iloc[row_loc-1]['P(G)3'])
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)3')] = P_C_func ( stuInfo.iloc[row_loc-1]['P(L)3'], stuInfo.iloc[row_loc-1]['P(S)3'], stuInfo.iloc[row_loc-1]['P(G)3'])
        
      
      if stuInfo.iloc[row_loc]['L4'] == 0.0:
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)4')] = stuInfo.iloc[row_loc-1]['P(L)4']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)4')] = stuInfo.iloc[row_loc-1]['P(C)4']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)4')] = stuInfo.iloc[row_loc-1]['P(S)4']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)4')] = stuInfo.iloc[row_loc-1]['P(G)4']
      else:
        if stuInfo.iloc[row_loc]['Correct']==0.0 and (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)4')] = P_S_func(stuInfo.iloc[row_loc-1]['P(L)4'], stuInfo.iloc[row_loc-1]['P(S)4'], stuInfo.iloc[row_loc-1]['P(G)4'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)4')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)4')]
        elif (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)4')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)4')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)4')] = P_G_func(stuInfo.iloc[row_loc-1]['P(L)4'], stuInfo.iloc[row_loc-1]['P(S)4'], stuInfo.iloc[row_loc-1]['P(G)4'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
        else:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)4')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)4')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)4')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)4')]
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)4')] = P_L_func ( stuInfo.iloc[row_loc]['Correct'], stuInfo.iloc[row_loc-1]['P(L)4'], stuInfo.iloc[row_loc-1]['P(S)4'], stuInfo.iloc[row_loc-1]['P(G)4'])
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)4')] = P_C_func ( stuInfo.iloc[row_loc-1]['P(L)4'], stuInfo.iloc[row_loc-1]['P(S)4'], stuInfo.iloc[row_loc-1]['P(G)4'])
        
     
      if stuInfo.iloc[row_loc]['L5'] == 0.0:
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)5')] = stuInfo.iloc[row_loc-1]['P(L)5']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)5')] = stuInfo.iloc[row_loc-1]['P(C)5']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)5')] = stuInfo.iloc[row_loc-1]['P(S)5']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)5')] = stuInfo.iloc[row_loc-1]['P(G)5']
      else:
        if stuInfo.iloc[row_loc]['Correct']==0.0 and (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)5')] = P_S_func(stuInfo.iloc[row_loc-1]['P(L)5'], stuInfo.iloc[row_loc-1]['P(S)5'], stuInfo.iloc[row_loc-1]['P(G)5'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)5')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)5')]
        elif (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)5')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)5')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)5')] = P_G_func(stuInfo.iloc[row_loc-1]['P(L)5'], stuInfo.iloc[row_loc-1]['P(S)5'], stuInfo.iloc[row_loc-1]['P(G)5'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
        else:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)5')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)5')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)5')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)5')]
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)5')] = P_L_func ( stuInfo.iloc[row_loc]['Correct'], stuInfo.iloc[row_loc-1]['P(L)5'], stuInfo.iloc[row_loc-1]['P(S)5'], stuInfo.iloc[row_loc-1]['P(G)5'])
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)5')] = P_C_func ( stuInfo.iloc[row_loc-1]['P(L)5'], stuInfo.iloc[row_loc-1]['P(S)5'], stuInfo.iloc[row_loc-1]['P(G)5'])
        
      
      if stuInfo.iloc[row_loc]['L6'] == 0.0:
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)6')] = stuInfo.iloc[row_loc-1]['P(L)6']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)6')] = stuInfo.iloc[row_loc-1]['P(C)6']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)6')] = stuInfo.iloc[row_loc-1]['P(S)6']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)6')] = stuInfo.iloc[row_loc-1]['P(G)6']
      else:
        if stuInfo.iloc[row_loc]['Correct']==0.0 and (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)6')] = P_S_func(stuInfo.iloc[row_loc-1]['P(L)6'], stuInfo.iloc[row_loc-1]['P(S)6'], stuInfo.iloc[row_loc-1]['P(G)6'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)6')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)6')]
        elif (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)6')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)6')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)6')] = P_G_func(stuInfo.iloc[row_loc-1]['P(L)6'], stuInfo.iloc[row_loc-1]['P(S)6'], stuInfo.iloc[row_loc-1]['P(G)6'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
        else:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)6')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)6')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)6')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)6')]
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)6')] = P_L_func ( stuInfo.iloc[row_loc]['Correct'], stuInfo.iloc[row_loc-1]['P(L)6'], stuInfo.iloc[row_loc-1]['P(S)6'], stuInfo.iloc[row_loc-1]['P(G)6'])
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)6')] = P_C_func ( stuInfo.iloc[row_loc-1]['P(L)6'], stuInfo.iloc[row_loc-1]['P(S)6'], stuInfo.iloc[row_loc-1]['P(G)6'])
        
      
      if stuInfo.iloc[row_loc]['L7'] == 0.0:
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)7')] = stuInfo.iloc[row_loc-1]['P(L)7']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)7')] = stuInfo.iloc[row_loc-1]['P(C)7']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)7')] = stuInfo.iloc[row_loc-1]['P(S)7']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)7')] = stuInfo.iloc[row_loc-1]['P(G)7']
      else:
        if stuInfo.iloc[row_loc]['Correct']==0.0 and (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)7')] = P_S_func(stuInfo.iloc[row_loc-1]['P(L)7'], stuInfo.iloc[row_loc-1]['P(S)7'], stuInfo.iloc[row_loc-1]['P(G)7'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)7')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)7')]
        elif (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)7')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)7')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)7')] = P_G_func(stuInfo.iloc[row_loc-1]['P(L)7'], stuInfo.iloc[row_loc-1]['P(S)7'], stuInfo.iloc[row_loc-1]['P(G)7'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
        else:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)7')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)7')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)7')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)7')]
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)7')] = P_L_func ( stuInfo.iloc[row_loc]['Correct'], stuInfo.iloc[row_loc-1]['P(L)7'], stuInfo.iloc[row_loc-1]['P(S)7'], stuInfo.iloc[row_loc-1]['P(G)7'])
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)7')] = P_C_func ( stuInfo.iloc[row_loc-1]['P(L)7'], stuInfo.iloc[row_loc-1]['P(S)7'], stuInfo.iloc[row_loc-1]['P(G)7'])
      
      if stuInfo.iloc[row_loc]['L8'] == 0.0:
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)8')] = stuInfo.iloc[row_loc-1]['P(L)8']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)8')] = stuInfo.iloc[row_loc-1]['P(C)8']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)8')] = stuInfo.iloc[row_loc-1]['P(S)8']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)8')] = stuInfo.iloc[row_loc-1]['P(G)8']
      else:
        if stuInfo.iloc[row_loc]['Correct']==0.0 and (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)8')] = P_S_func(stuInfo.iloc[row_loc-1]['P(L)8'], stuInfo.iloc[row_loc-1]['P(S)8'], stuInfo.iloc[row_loc-1]['P(G)8'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)8')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)8')]
        elif (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)8')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)8')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)8')] = P_G_func(stuInfo.iloc[row_loc-1]['P(L)8'], stuInfo.iloc[row_loc-1]['P(S)8'], stuInfo.iloc[row_loc-1]['P(G)8'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
        else:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)8')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)8')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)8')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)8')]
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)8')] = P_L_func ( stuInfo.iloc[row_loc]['Correct'], stuInfo.iloc[row_loc-1]['P(L)8'], stuInfo.iloc[row_loc-1]['P(S)8'], stuInfo.iloc[row_loc-1]['P(G)8'])
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)8')] = P_C_func ( stuInfo.iloc[row_loc-1]['P(L)8'], stuInfo.iloc[row_loc-1]['P(S)8'], stuInfo.iloc[row_loc-1]['P(G)8'])  
      
      if stuInfo.iloc[row_loc]['L9'] == 0.0:
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)9')] = stuInfo.iloc[row_loc-1]['P(L)9']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)9')] = stuInfo.iloc[row_loc-1]['P(C)9']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)9')] = stuInfo.iloc[row_loc-1]['P(S)9']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)9')] = stuInfo.iloc[row_loc-1]['P(G)9']
      else:
        if stuInfo.iloc[row_loc]['Correct']==0.0 and (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)9')] = P_S_func(stuInfo.iloc[row_loc-1]['P(L)9'], stuInfo.iloc[row_loc-1]['P(S)9'], stuInfo.iloc[row_loc-1]['P(G)9'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)9')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)9')]
        elif (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)9')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)9')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)9')] = P_G_func(stuInfo.iloc[row_loc-1]['P(L)9'], stuInfo.iloc[row_loc-1]['P(S)9'], stuInfo.iloc[row_loc-1]['P(G)9'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
        else:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)9')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)9')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)9')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)9')]
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)9')] = P_L_func ( stuInfo.iloc[row_loc]['Correct'], stuInfo.iloc[row_loc-1]['P(L)9'], stuInfo.iloc[row_loc-1]['P(S)9'], stuInfo.iloc[row_loc-1]['P(G)9'])
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)9')] = P_C_func ( stuInfo.iloc[row_loc-1]['P(L)9'], stuInfo.iloc[row_loc-1]['P(S)9'], stuInfo.iloc[row_loc-1]['P(G)9']) 
    
      if stuInfo.iloc[row_loc]['L10'] == 0.0:
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)10')] = stuInfo.iloc[row_loc-1]['P(L)10']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)10')] = stuInfo.iloc[row_loc-1]['P(C)10']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)10')] = stuInfo.iloc[row_loc-1]['P(S)10']
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)10')] = stuInfo.iloc[row_loc-1]['P(G)10']
      else:
        if stuInfo.iloc[row_loc]['Correct']==0.0 and (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)10')] = P_S_func(stuInfo.iloc[row_loc-1]['P(L)10'], stuInfo.iloc[row_loc-1]['P(S)10'], stuInfo.iloc[row_loc-1]['P(G)10'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)10')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)10')]
        elif (row_loc+1)%6!=5 and (row_loc+1)%6!=0:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)10')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)10')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)10')] = P_G_func(stuInfo.iloc[row_loc-1]['P(L)10'], stuInfo.iloc[row_loc-1]['P(S)10'], stuInfo.iloc[row_loc-1]['P(G)10'], stuInfo.iloc[row_loc+1]['Correct'], stuInfo.iloc[row_loc+2]['Correct'])
        else:
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(G)10')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(G)10')]
            stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(S)10')] = stuInfo.iloc[row_loc-1,stuInfo.columns.get_loc('P(S)10')]
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(L)10')] = P_L_func ( stuInfo.iloc[row_loc]['Correct'], stuInfo.iloc[row_loc-1]['P(L)10'], stuInfo.iloc[row_loc-1]['P(S)10'], stuInfo.iloc[row_loc-1]['P(G)10'])
        stuInfo.iloc[row_loc,stuInfo.columns.get_loc('P(C)10')] = P_C_func ( stuInfo.iloc[row_loc-1]['P(L)10'], stuInfo.iloc[row_loc-1]['P(S)10'], stuInfo.iloc[row_loc-1]['P(G)10'])
    
    if row_loc < rows: row_loc += 1
    
  

  
  stuInfo.to_csv('stu' + str(Student) +'.csv', index=True, header=True, float_format='%.3f')
  print('Saved %s rows' %(rows))
  


stuInfo

