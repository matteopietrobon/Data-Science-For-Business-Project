# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:24:56 2017

@author: Teo
"""

import pandas as pd

seed = 123

df_complete = pd.read_json(path_or_buf = 'data/reviews_Digital_Music_5.json', lines=True)
df1 = pd.DataFrame(df_complete[['asin', 'overall','reviewText']])
del df_complete

df_complete = df = pd.read_pickle('amazon_step2_unseen.pkl')
df2 = pd.DataFrame(df_complete[['asin','reviewText']])
del df_complete


#JOIN TRAIN AND TEST SET TO PRE-PROCESS THEM TOGETHER
df = pd.concat([df1.ix[:,[0,2]], df2])
df.reset_index(inplace=True)

drop_indexes = []
for i in range(1,df.shape[0]):
    if(isinstance(df.ix[i,2],str) == False):
    
        drop_indexes.append(i)
        
#THESE ELEMENTS are NaNs I drop them
dff = df.drop(df.index[drop_indexes])
#%%

df_refined=[]
for item in dff.ix[:,2]:
    df_refined.append (item.replace('\r',' ').replace('/n',' ').replace('.',' ')\
                           .replace(',',' ').replace('(',' ').replace(')',' ')\
                           .replace("'s",' ').replace('"',' ').replace('!',' ')\
                           .replace('?',' ').replace("'",' ').replace('>',' ')\
                           .replace('$',' ').replace('-',' ').replace(';',' ')\
                           .replace(':',' ').replace('/',' ').replace('#',' '))
    
from gensim import corpora, models, matutils
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk import download

download('wordnet')

tester = 1
lemmatizer = WordNetLemmatizer()
documents = df_refined

# removing stopwords
documents_no_stop = [[word for word in document.lower().split() if word not in STOPWORDS]
         for document in documents]

del documents

# remove words that appear only once
from collections import defaultdict
threshold = 1 # frequency threshold
frequency = defaultdict(int)
for text in documents_no_stop:
    for token in text:
        frequency[token] += 1

documents_no_stop_no_unique = [[token for token in text if frequency[token] > threshold] 
                               for text in documents_no_stop]

del documents_no_stop

# remove all numerics and tokens with numbers
import re
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))
documents_no_stop_no_unique_no_numeric = [[token for token in text if not (hasNumbers(token)) ] 
                                          for text in documents_no_stop_no_unique]

del documents_no_stop_no_unique

# lemmattizing tokens (better than stemming by taking word context into account)
documents_no_stop_no_unique_no_numeric_lemmatize = [[lemmatizer.lemmatize(token) for token in text] 
                                                    for text in documents_no_stop_no_unique_no_numeric]

del documents_no_stop_no_unique_no_numeric

import enchant
eng_dic = enchant.Dict("en_US")

# remove non-english words
documents_no_stop_no_unique_no_numeric_lemmatize_english = [[token for token in text if (eng_dic.check(token)) ] 
                                                            for text in documents_no_stop_no_unique_no_numeric_lemmatize]

del documents_no_stop_no_unique_no_numeric_lemmatize

# create ready corpus
ready_corpus = documents_no_stop_no_unique_no_numeric_lemmatize_english

# build the dictionary and store it to disc for future use
dictionary = corpora.Dictionary(ready_corpus)
print(dictionary)

# convert the corpus into bag of words 
corpus_bow = [dictionary.doc2bow(comment) for comment in ready_corpus]

tfidf_transformer = models.TfidfModel(corpus_bow, normalize=True)

# apply tfidf transformation to the bow corpus
corpus_tfidf = tfidf_transformer [corpus_bow]

# convert to a sparse and compatible format for dimensionality reduction using sklearn
sparse_corpus_tfidf = matutils.corpus2csc(corpus_tfidf)
sparse_corpus_tfidf_transpose = sparse_corpus_tfidf.transpose()

X_train = sparse_corpus_tfidf_transpose[:df1.shape[0],:]
X_test = sparse_corpus_tfidf_transpose[df1.shape[0]:,:]
y_train = df1.ix[:,1]

#%%

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(C=3.6)

print('Fitting Logistic Regression')
log_reg.fit(X_train,y_train)

y_cap = log_reg.predict(X_test)

#%%
import numpy as np
#SAVE FILE AS CSV
data = np.column_stack((df2['asin'],y_cap))
np.savetxt('predictions/step3.csv',data, fmt="%s", delimiter = ",")