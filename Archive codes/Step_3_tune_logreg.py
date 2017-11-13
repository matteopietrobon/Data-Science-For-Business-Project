# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:28:31 2017

@author: Teo
"""

import pandas as pd

seed = 123

df_complete = pd.read_json(path_or_buf = 'data/reviews_Digital_Music_5.json', lines=True)
df = pd.DataFrame(df_complete[['asin', 'overall','reviewText']])
df = df.ix[:30000,:]
del df_complete
five_stars = df['overall']

# We are not removing all the special characters.
# NLTK will handle the special characters within the functions, they are used to
# assess what words are influenced by negative terms (e.g. 'I do not like it. It is bad!' 
# will consider the 'like' as negated because it is between the 'not' and the '.',
# 'bad' will be considered as normal, for the sentence 'I like it. It is not bad!'
# it will do the opposite). We will only keep the special characters that NLTK uses: ^[.:;!?]
# Since sometimes there was no space after them we are adding it.
df_refined=[]
for item in df.ix[:,2]:
    df_refined.append (item.replace('^','^ ').replace('[','[ ').replace('.','. ')\
                           .replace(',',', ').replace(':',': ').replace(';','; ')\
                           .replace("]",'] ').replace('!','! ').replace('?','? ')\
                           .replace('$',' ').replace("<",' ').replace('>',' ')\
                           .replace('-',' ').replace('/',' ').replace('#',' '))
    
from nltk.stem import WordNetLemmatizer

tester = 1
lemmatizer = WordNetLemmatizer()
documents = df_refined

# Tokenize words
from nltk.tokenize import word_tokenize
from nltk import download
download('punkt')

documents_tokenized = [word_tokenize(document) for document in documents]

# lemmattizing tokens (better than stemming by taking word context into account)
documents_tokenized_lemmatized = [[lemmatizer.lemmatize(token) for token in text] 
                                                    for text in documents_tokenized]

from nltk.sentiment.util import mark_negation

documents_tokenized_lemmatized_negated = [mark_negation(document) for document in documents_tokenized_lemmatized]

ready_corpus=documents_tokenized_lemmatized_negated

download('opinion_lexicon')
from nltk.corpus import opinion_lexicon

# we consider only sentiment words, opinion_lexicon icludes already mispelled sentiment words,
# so we did not use the enchant library this time.
sentiment_words= opinion_lexicon.words()
sentiment_words_negated= [word+'_NEG' for word in sentiment_words]

sentiment_features=sentiment_words+sentiment_words_negated

from gensim import corpora, models, matutils
# build the dictionary
dictionary = corpora.Dictionary(ready_corpus)
print(dictionary)

from nltk.sentiment import SentimentAnalyzer

sentiment_analizer= SentimentAnalyzer()
list_all_words=sentiment_analizer.all_words(ready_corpus)

used_sentiment_words=list(set(sentiment_features).intersection(list_all_words))

# determine which tokens to keep
good_ids = [dictionary.token2id[word] for word in used_sentiment_words]
        
dictionary.filter_tokens(good_ids=good_ids)
dictionary.compactify()
print(dictionary)
# convert the corpus into bag of words 
corpus_bow = [dictionary.doc2bow(comment) for comment in ready_corpus]

tfidf_transformer = models.TfidfModel(corpus_bow, normalize=True)

# apply tfidf transformation to the bow corpus
corpus_tfidf = tfidf_transformer [corpus_bow]

# convert to a sparse and compatible format for dimensionality reduction using sklearn
sparse_corpus_tfidf = matutils.corpus2csc(corpus_tfidf)
sparse_corpus_tfidf_transpose = sparse_corpus_tfidf.transpose()

from sklearn.model_selection import train_test_split
train_tfidf, test_tfidf, train_category, test_category = train_test_split(sparse_corpus_tfidf_transpose,\
                                                                          df.ix[:,1], test_size = 0.2, random_state = seed)

#print('Starting dimensionality reduction')
#
## reduce dimensions
#from sklearn.decomposition import KernelPCA
#
#reducer= KernelPCA(n_components = 150, kernel="cosine", random_state=seed)
#corpus_train_tfidf_kpca = reducer.fit_transform(train_tfidf)
#corpus_test_tfidf_kpca = reducer.transform(test_tfidf)
#
#print('Finished dimensionality reduction')

from sklearn.metrics import confusion_matrix
import numpy as np

def our_scoring_function(y_true,y_pred):
    penalties=np.zeros([5,5])
    for i in np.arange(5):
        for j in np.arange(5):
            penalties[i,j]= -np.abs(i-j)
    conf_mat = confusion_matrix(y_true, y_pred,labels=np.arange(1,6))/np.shape(y_true)[0]
    return np.sum(penalties*conf_mat)
    

from sklearn.metrics import make_scorer
our_scoring= make_scorer(our_scoring_function)

X_train = train_tfidf
X_test = test_tfidf
y_train = train_category
y_test = test_category


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
             
#Initialize K-Fold for cross validation
K = 5
kfold = KFold(n_splits=K, random_state=seed)

#%%
# LOGISTIC REGRESSION
estimators = []
estimators.append(('normalizer', Normalizer()))
estimators.append(('log_reg', LogisticRegression()))
log_reg_pipe1 = Pipeline(estimators)

penalties = np.logspace(0,1,4)


parameters = {
        'log_reg__C' : penalties
}
estimator_log_reg = GridSearchCV(log_reg_pipe1, parameters,scoring=our_scoring, cv=kfold)
                  
# evaluate the grid search and print best classifier
estimator_log_reg.fit(X_train,y_train)

alphas = [x['log_reg__C'] for x in estimator_log_reg.cv_results_['params']]
means = [x for x in estimator_log_reg.cv_results_['mean_test_score']]
stds = [x for x in estimator_log_reg.cv_results_['std_test_score']]

plt.figure(figsize=(8, 6))
plt.errorbar(alphas, means, stds, fmt='o', lw=1)
plt.plot(alphas, means)
plt.xlabel('penalty')
plt.ylabel('mean score')
plt.title('Logistic Regression')
plt.xscale('log')
plt.show()

print('\nBest penalty for Logistic Regression --->   ',estimator_log_reg.best_params_['log_reg__C'])

log_reg_pipe1.set_params(log_reg__C = estimator_log_reg.best_params_['log_reg__C'])
log_reg_pipe1.fit(X_train,y_train)
score_logreg=our_scoring_function(y_test,log_reg_pipe1.predict(X_test))
print ('\nLogistic regression test score --->   ', score_logreg)