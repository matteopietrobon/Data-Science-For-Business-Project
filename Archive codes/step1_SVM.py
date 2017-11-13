
import pandas as pd

seed = 123

df_complete = pd.read_json(path_or_buf = 'data/amazon_step1.json', lines=True)
df = pd.DataFrame(df_complete[['asin', 'category','reviewText']])
df = df.ix[:5000,:]
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



#%%

# convert to a sparse and compatible format for dimensionality reduction using sklearn
sparse_corpus_tfidf = matutils.corpus2csc(corpus_tfidf)
sparse_corpus_tfidf_transpose = sparse_corpus_tfidf.transpose()


train_tfidf, test_tfidf, train_category, test_category = train_test_split(sparse_corpus_tfidf_transpose, united.ix[:,1], test_size = 0.2, random_state = seed)

#%%

print('starting dimensionality reduction')
# reduce dimensions
from sklearn.decomposition import KernelPCA
reducer= KernelPCA(n_components = 30 , kernel="cosine", random_state=seed)
corpus_train_tfidf_kpca = reducer.fit_transform(train_tfidf)
corpus_test_tfidf_kpca = reducer.transform(test_tfidf)

corpus_train_tfidf_reduced=corpus_train_tfidf_kpca
corpus_test_tfidf_reduced =corpus_test_tfidf_kpca

#print('starting tsne')
#from sklearn.manifold import TSNE
#reducer = TSNE(n_components = 30, learning_rate=1000.0, n_iter=1000, metric='cosine')
#corpus_train_tfidf_reduced = reducer.fit_transform(corpus_train_tfidf_kpca)
#corpus_test_tfidf_reduced = reducer.transform(corpus_test_tfidf_kpca)


#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
                  


K = 5
kfold = KFold(n_splits=K, random_state=seed)

# LINEAR KERNEL
estimators = []
estimators.append(('normalizar', preprocessing.Normalizer()))
estimators.append(('svm_linear_clf', SVC()))
svm_linear_pipe1 = Pipeline(estimators)
svm_linear_pipe1.set_params(svm_linear_clf__kernel='linear', svm_linear_clf__gamma='auto')

penalties = np.logspace(-5,3,9)


parameters = {
        'svm_linear_clf__C' : penalties
}
estimator_svm_linear = GridSearchCV(svm_linear_pipe1, parameters, cv=kfold)
                  
# evaluate the grid search and print best classifier
estimator_svm_linear.fit(corpus_train_tfidf_reduced, train_category)

alphas = [x['svm_linear_clf__C'] for x in estimator_svm_linear.cv_results_['params']]
means = [x for x in estimator_svm_linear.cv_results_['mean_test_score']]
stds = [x for x in estimator_svm_linear.cv_results_['std_test_score']]

plt.figure(figsize=(8, 6))
plt.errorbar(alphas, means, stds, fmt='o', lw=1)
plt.plot(alphas, means)
plt.xlabel('penalty')
plt.ylabel('mean accuracy')
plt.title('SVC, linear kernel')
plt.xscale('log')
plt.show()

print('\nBest penalty --->   ',estimator_svm_linear.best_params_['svm_linear_clf__C'])

svm_linear_pipe1.set_params(svm_linear_clf__C = estimator_svm_linear.best_params_['svm_linear_clf__C'])
svm_linear_pipe1.fit(corpus_train_tfidf_reduced, train_category)
accuracy_b = svm_linear_pipe1.score(corpus_test_tfidf_reduced, test_category)
print ('\nTest score --->   ', accuracy_b)



# GAUSSIAN KERNEL
estimators = []
estimators.append(('normalizar', preprocessing.Normalizer()))
estimators.append(('svm_gaussian_clf', SVC()))
svm_gaussian_pipe1 = Pipeline(estimators)
svm_gaussian_pipe1.set_params(svm_gaussian_clf__kernel='rbf', svm_gaussian_clf__gamma='auto')

penalties = np.logspace(-5,3,9)


parameters = {
        'svm_gaussian_clf__C' : penalties
}
estimator_svm_gaussian = GridSearchCV(svm_gaussian_pipe1, parameters, cv=kfold)
                  
# evaluate the grid search and print best classifier
estimator_svm_gaussian.fit(corpus_train_tfidf_reduced, train_category)

alphas = [x['svm_gaussian_clf__C'] for x in estimator_svm_gaussian.cv_results_['params']]
means = [x for x in estimator_svm_gaussian.cv_results_['mean_test_score']]
stds = [x for x in estimator_svm_gaussian.cv_results_['std_test_score']]

plt.figure(figsize=(8, 6))
plt.errorbar(alphas, means, stds, fmt='o', lw=1)
plt.plot(alphas, means)
plt.xlabel('penalty')
plt.ylabel('mean accuracy')
plt.title('SVC, gaussian kernel')
plt.xscale('log')
plt.show()

print('\nBest penalty --->   ',estimator_svm_gaussian.best_params_['svm_gaussian_clf__C'])

svm_gaussian_pipe1.set_params(svm_gaussian_clf__C = estimator_svm_gaussian.best_params_['svm_gaussian_clf__C'])
svm_gaussian_pipe1.fit(corpus_train_tfidf_reduced, train_category)
accuracy_b = svm_gaussian_pipe1.score(corpus_test_tfidf_reduced, test_category)
print ('\nTest score --->   ', accuracy_b)



# LOGISTIC REGRESSION
estimators = []
estimators.append(('normalizar', preprocessing.Normalizer()))
estimators.append(('log_reg', LogisticRegression()))
log_reg_pipe1 = Pipeline(estimators)
log_reg_pipe1.set_params()

penalties = np.logspace(-5,3,9)


parameters = {
        'log_reg__C' : penalties
}
estimator_log_reg = GridSearchCV(log_reg_pipe1, parameters, cv=kfold)
                  
# evaluate the grid search and print best classifier
estimator_log_reg.fit(corpus_train_tfidf_reduced, train_category)

alphas = [x['log_reg__C'] for x in estimator_log_reg.cv_results_['params']]
means = [x for x in estimator_log_reg.cv_results_['mean_test_score']]
stds = [x for x in estimator_log_reg.cv_results_['std_test_score']]

plt.figure(figsize=(8, 6))
plt.errorbar(alphas, means, stds, fmt='o', lw=1)
plt.plot(alphas, means)
plt.xlabel('penalty')
plt.ylabel('mean accuracy')
plt.title('Logistic Regression')
plt.xscale('log')
plt.show()

print('\nBest penalty --->   ',estimator_log_reg.best_params_['log_reg__C'])

log_reg_pipe1.set_params(log_reg__C = estimator_log_reg.best_params_['log_reg__C'])
log_reg_pipe1.fit(corpus_train_tfidf_reduced, train_category)
accuracy_c = log_reg_pipe1.score(corpus_test_tfidf_reduced, test_category)
print ('\nTest score --->   ', accuracy_c)




