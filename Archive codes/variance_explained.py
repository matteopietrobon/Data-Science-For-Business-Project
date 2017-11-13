# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:37:53 2017

@author: Teo
"""

import pandas as pd

seed = 123

df_complete = pd.read_json(path_or_buf = 'data/amazon_step1.json', lines=True)
df = pd.DataFrame(df_complete[['asin', 'category','reviewText']])
df = df.ix[:10000,:]
del df_complete

#%%
df_refined=[]
for item in df.ix[:,2]:
    df_refined.append (item.replace('\r',' ').replace('/n',' ').replace('.',' ')\
                           .replace(',',' ').replace('(',' ').replace(')',' ')\
                           .replace("'s",' ').replace('"',' ').replace('!',' ')\
                           .replace('?',' ').replace("'",' ').replace('>',' ')\
                           .replace('$',' ').replace('-',' ').replace(';',' ')\
                           .replace(':',' ').replace('/',' ').replace('#',' '))

# enchant only works on 32-bit python
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk import download

download('wordnet')

tester = 1
lemmatizer = WordNetLemmatizer()
documents = df_refined
print ('original: ',documents[tester], '\n')


#%%
# we should check that STOPWORDS are right for us, mayve more, maybe less
documents_no_stop = [[word for word in document.lower().split() if word not in STOPWORDS]
         for document in documents]

print ('tokenize and remove stop words: ',documents_no_stop[tester], '\n')
del documents
#%%

# remove words that appear only once
from collections import defaultdict
threshold = 1 # frequency threshold
frequency = defaultdict(int)
for text in documents_no_stop:
    for token in text:
        frequency[token] += 1

documents_no_stop_no_unique = [[token for token in text if frequency[token] > threshold] 
                               for text in documents_no_stop]

print ('remove unique words: ',documents_no_stop_no_unique[tester], '\n')
del documents_no_stop

#%%

# remove all numerics and tokens with numebrs, is this a good idea?
import re
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))
documents_no_stop_no_unique_no_numeric = [[token for token in text if not (hasNumbers(token)) ] 
                                          for text in documents_no_stop_no_unique]

print ('remove numerics: ',documents_no_stop_no_unique_no_numeric[tester], '\n')
del documents_no_stop_no_unique


#%%

# lemmattizing tokens (better than stemming by taking word context into account)
documents_no_stop_no_unique_no_numeric_lemmatize = [[lemmatizer.lemmatize(token) for token in text] 
                                                    for text in documents_no_stop_no_unique_no_numeric]

print ('lemmatize: ',documents_no_stop_no_unique_no_numeric_lemmatize[tester], '\n')
del documents_no_stop_no_unique_no_numeric


#%%
import enchant
eng_dic = enchant.Dict("en_US")

# remove non-english words
documents_no_stop_no_unique_no_numeric_lemmatize_english = [[token for token in text if (eng_dic.check(token)) ] 
                                                            for text in documents_no_stop_no_unique_no_numeric_lemmatize]

print ('no english: ',documents_no_stop_no_unique_no_numeric_lemmatize_english[tester], '\n')
del documents_no_stop_no_unique_no_numeric_lemmatize


#%%

# create ready corpus
ready_corpus = documents_no_stop_no_unique_no_numeric_lemmatize_english
#ready_corpus = documents_no_stop_no_unique_no_numeric_lemmatize
print (len(ready_corpus))

# build the dictionary and store it to disc for future use
dictionary = corpora.Dictionary(ready_corpus)
dictionary.save('saved/step1_dict.dict') 
print(dictionary)

#%%

# convert the corpus into bag of words 
corpus_bow = [dictionary.doc2bow(comment) for comment in ready_corpus]

# save to disk for future use
corpora.MmCorpus.serialize('saved/step1_bow.mm', corpus_bow)
i = 0
for doc in corpus_bow:
    print(doc)
    print('')
    i+=1
    if i > 5: break

#%%

import os
from gensim import corpora, models, matutils

# load bow representation of corpus
if (os.path.exists('saved/step1_bow.mm')):
    corpus_bow = corpora.MmCorpus('saved/step1_bow.mm')
    print("Load files generated from previous parts")
else:
    print("Please run previous parts to generate data sets")
    
#%%
from sklearn.model_selection import train_test_split

# should we put only train here?
tfidf_transformer = models.TfidfModel(corpus_bow, normalize=True)

# apply tfidf transformation to the bow corpus
corpus_tfidf = tfidf_transformer [corpus_bow]

print(len(corpus_tfidf.corpus))
for doc in corpus_tfidf:
    print(doc)
    break

# convert to a sparse and compatible format for dimensionality reduction using sklearn
sparse_corpus_tfidf = matutils.corpus2csc(corpus_tfidf)
sparse_corpus_tfidf_transpose = sparse_corpus_tfidf.transpose()



#%%

print('starting dimensionality reduction')
# reduce dimensions
from sklearn.decomposition import KernelPCA
import numpy as np
reducer= KernelPCA(n_components = 700 , kernel="cosine", random_state=seed)
corpus_tfidf_kpca = reducer.fit_transform(sparse_corpus_tfidf_transpose)

explained_variance = np.var(corpus_tfidf_kpca, axis=0)
#total_variance = np.var(train_tfidf, axis=0)
#explained_variance_ratio = explained_variance / np.sum(total_variance)
explained_variance_ratio = explained_variance / np.sum(explained_variance)

cum_explained_variance=np.cumsum(explained_variance_ratio)

#%%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(explained_variance_ratio)

#plt.figure()
#plt.plot(cum_explained_variance)
