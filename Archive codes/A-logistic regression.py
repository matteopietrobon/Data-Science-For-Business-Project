#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:40:55 2017

@author: mpietrob
"""


import pandas as pd
from sklearn.model_selection import train_test_split

seed = 123

df_complete = pd.read_json(path_or_buf = 'amazon_step1.json', lines=True)
df = pd.DataFrame(df_complete[['asin', 'category','reviewText']])
df = df.ix[:20000,:]
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

del df
#%%
united_refined=[]
for item in united.ix[:,2]:
    united_refined.append (item.replace('\r',' ').replace('/n',' ').replace('.',' ')\
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
documents = united_refined
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
dictionary.save('step1_dict.dict') 
print(dictionary)

#%%

# convert the corpus into bag of words 
corpus_bow = [dictionary.doc2bow(comment) for comment in ready_corpus]

# save to disk for future use
corpora.MmCorpus.serialize('step1_bow.mm', corpus_bow)
i = 0
for doc in corpus_bow:
    print(doc)
    print('')
    i+=1
    if i > 5: break

#%%

import os
from gensim import corpora, models, similarities, matutils

# load bow representation of corpus
if (os.path.exists('step1_bow.mm')):
    corpus_bow = corpora.MmCorpus('step1_bow.mm')
    print("Load files generated from previous parts")
else:
    print("Please run previous parts to generate data sets")
    
#%%
# divide in test and train sets
train_corpus, test_corpus, train_category, test_category = train_test_split(corpus_bow, united.ix[:,1], test_size = 0.2, random_state = seed)

# should we put only train here?
tfidf_transformer = models.TfidfModel(corpus_bow, normalize=True)

# apply tfidf transformation to the bow corpus
train_tfidf = tfidf_transformer [train_corpus]

print(len(train_tfidf.corpus))
for doc in train_tfidf:
    print(doc)
    break

#%%


# saving the model to disk for future use
corpora.MmCorpus.serialize('train_tfidf.mm', train_tfidf)
# convert to a sparse and compatible format for dimensionality reduction using sklearn
sparse_train_corpus_tfidf = matutils.corpus2csc(train_tfidf)
sparse_train_corpus_tfidf_transpose = sparse_train_corpus_tfidf.transpose()

#%%

# visualize the tf-idf corpus using kernel PCA
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np
kpca2 = KernelPCA(n_components = 200 , kernel="cosine", random_state=seed)
corpus_train_tfidf_kpca = kpca2.fit_transform(sparse_train_corpus_tfidf_transpose)

explained_variance = np.var(corpus_train_tfidf_kpca, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)

#%%
plt.figure(1)
plt.plot(explained_variance_ratio)
plt.title('Explained variance')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(explained_variance_ratio))
plt.title('Cumulative explained variance')
plt.show()

#%%
# TSE? too many features
#sklearn.manifold.TSNE(n_components=2, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0,metric='cosine', random_state=seed)



#%%
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
#%%

N = 11000
log_reg = LogisticRegression(C=1000, random_state = seed)
log_reg.fit(corpus_train_tfidf_kpca[:N:], train_category[:N])




y_cap = log_reg.predict(corpus_train_tfidf_kpca[N:,:])
justosee = pd.DataFrame(train_category[N:])
justosee['Predicted'] = y_cap
justosee['Prob'] = log_reg.predict_proba(corpus_train_tfidf_kpca[N:,:])[:,1]

#%%

# get the names of the categories for the plots
Labels=train_category.unique()
# Compute confusion matrix
cnf_matrix = confusion_matrix(train_category[N:], y_cap, labels=Labels)

# Plot normalized confusion matrix
plt.figure(2)
plot_confusion_matrix(cnf_matrix,Labels, normalize=False,
                      title='Normalized confusion matrix')

plt.show()
#a = y_cap == 1 
#c = y_train[N:] == 1
#b = y_cap == y_train[N:]

#print('My score for Logistic Regression:', sum(a & b)/sum(c))
