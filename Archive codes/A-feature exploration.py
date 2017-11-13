#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:07:11 2017

@author: mpietrob
"""


import pandas as pd
from sklearn.model_selection import train_test_split

seed = 123

df_complete = pd.read_json(path_or_buf = 'amazon_step1.json', lines=True)
df = pd.DataFrame(df_complete[['asin', 'category','reviewText']])
#df = df.ix[:2000,:]

categories= pd.get_dummies(df['category'])
print(categories.sum())



del df_complete
#%%
#df.isnull().any().any()

united = df.groupby(['asin', 'category'])['reviewText'].apply(' '.join).reset_index()

old = 0

indexes = []

for i,row in united.iterrows():
    
    if(old == row['asin']):
        
        
        indexes.append(i)
        indexes.append(i-1)
        
    old = row['asin']

#print(united['asin'].value_counts()) TO CHECK MOST FREQUENT OBJECTS

united.drop(united.index[[indexes]], inplace = True)
categories= pd.get_dummies(df['category'])
print(categories.sum())

del df
