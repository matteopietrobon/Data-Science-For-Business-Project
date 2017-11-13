# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:04:01 2017

@author: Teo
"""

import pandas as pd

seed = 123

df_complete = pd.read_json(path_or_buf = 'data/reviews_Digital_Music_5.json', lines=True)
df = pd.DataFrame(df_complete[['asin', 'overall','reviewText']])
df = df.ix[:1000,:]
del df_complete
five_stars = df['overall']


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
scores=[analyzer.polarity_scores(sentence) for sentence in df.ix[:,2]]

import numpy as np
scores_array=np.array([(comment['neg'],comment['neu'],comment['pos']) for comment in scores])


# visualize the tf-idf corpus using kernel PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#RENAMED FOR EASE
X = scores_array

#CREATE DICTIONARY TO ASSIGN COLORS
categories = five_stars.unique()

#REINDEX OUTPUT TO COMPARE WITH LABELS
reindexed = five_stars.reset_index()

#CREATE DICTIONARY TO ASSIGN VALUES TO OCCURRENCE VECTOR AND COLORS
together = zip(categories,np.arange(0,5))
locator = dict(together)
occurred =  np.zeros(5)

reds  = [0, 0.6, 0.86,   1,  0]
greens =[1, 0.3, 0.67, 0.8,  0]
blues = [0,   0,  0.3,   1,  1]   

markers = ['o', 'v', '+', '<', '>']
colors = pd.DataFrame(reds, columns = ['Reds'])
colors['Greens'] = greens
colors['Blues'] = blues

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')

for i in range(0,len(five_stars)):

    index = locator[reindexed.ix[i,1]]
    
    if(occurred[locator[reindexed.ix[i,1]]] == 0):
        
        occurred[locator[reindexed.ix[i,1]]] = 1
        ax.scatter(X[i,0], X[i,1], X[i,2], color = colors.ix[index,:], label = reindexed.ix[i,1], marker=markers[index])
        
    else:
        #NO LABEL IF ALREADY EXISTING
        ax.scatter(X[i,0], X[i,1], X[i,2], color = colors.ix[index,:],marker=markers[index])
        
plt.legend()
plt.title("3D Data Projection")