
import pandas as pd

seed = 123

df_complete = pd.read_json(path_or_buf = 'data/amazon_step1.json', lines=True)
df = pd.DataFrame(df_complete[['asin', 'category','reviewText']])
df = df.ix[:1000,:]
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



#%%

# convert to a sparse and compatible format for dimensionality reduction using sklearn
sparse_corpus_tfidf = matutils.corpus2csc(corpus_tfidf)
sparse_corpus_tfidf_transpose = sparse_corpus_tfidf.transpose()


train_tfidf, test_tfidf, train_category, test_category = train_test_split(sparse_corpus_tfidf_transpose, df.ix[:,1], test_size = 0.2, random_state = seed)

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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier



K = 5
kfold = KFold(n_splits=K, random_state=seed)


n_tried=25
depths=np.arange(1,n_tried)

estimators = []
estimators.append(('Normalizar', preprocessing.Normalizer()))
estimators.append(('tree_cla', tree.DecisionTreeClassifier(random_state=seed)))
cla_tree_pipe1 = Pipeline(estimators)

parameters = {
        'tree_cla__max_depth' : depths
}
estimator_treecla = GridSearchCV(cla_tree_pipe1, parameters, cv=kfold)
                  
# evaluate the grid search and print best classifier
estimator_treecla.fit(corpus_train_tfidf_reduced, train_category)

alphas = [x['tree_cla__max_depth'] for x in estimator_treecla.cv_results_['params']]
means = [x for x in estimator_treecla.cv_results_['mean_test_score']]
stds = [x for x in estimator_treecla.cv_results_['std_test_score']]

plt.figure(figsize=(8, 6))
plt.errorbar(alphas, means, stds, fmt='o', lw=1)
plt.plot(alphas, means)
plt.xlabel('max depth')
plt.ylabel('mean cross validation R2')
plt.title('Decision Tree')
plt.show()

print('\nBest max depth --->   ',estimator_treecla.best_params_['tree_cla__max_depth'])

cla_tree_pipe1.set_params(tree_cla__max_depth = estimator_treecla.best_params_['tree_cla__max_depth'])
cla_tree_pipe1.fit(corpus_train_tfidf_reduced, train_category)
R2_b = cla_tree_pipe1.score(corpus_test_tfidf_reduced, test_category)
print ('\nTest score --->   ', R2_b)
print ('\nFeature importances:\n')
importances = cla_tree_pipe1.named_steps['tree_cla'].feature_importances_
indices = np.argsort(importances)[::-1]
for i in indices:
    print('%.3f' %importances[i],' <---  feature ',i)
# Plot the feature importances
plt.figure(figsize=(8, 6))
plt.title("Feature importances")
sns.barplot(indices, y=importances[indices])
plt.show()


estimators = []
estimators.append(('Normalizar', preprocessing.Normalizer()))
estimators.append(('bag_cla', BaggingClassifier()))
cla_bag_pipe1 = Pipeline(estimators)
cla_bag_pipe1.set_params(bag_cla__base_estimator=tree.DecisionTreeClassifier(max_depth=estimator_treecla.best_params_['tree_cla__max_depth']),\
                         bag_cla__n_estimators=500, bag_cla__random_state=seed)

cla_bag_pipe1.fit(corpus_train_tfidf_reduced, train_category)
R2_c=cla_bag_pipe1.score(corpus_test_tfidf_reduced, test_category)
print ('\nBagging Test score --->   ', R2_c)


estimators = []
estimators.append(('Normalizar', preprocessing.Normalizer()))
estimators.append(('for_cla', RandomForestClassifier()))
cla_for_pipe1 = Pipeline(estimators)
cla_for_pipe1.set_params(for_cla__n_estimators=100,\
                         for_cla__max_depth=estimator_treecla.best_params_['tree_cla__max_depth'], for_cla__random_state=seed)

parameters = {
        'for_cla__max_features' : range(5,15)
}
estimator_forcla = GridSearchCV(cla_for_pipe1, parameters, cv=kfold)
                  
# evaluate the grid search and print best classifier
estimator_forcla.fit(corpus_train_tfidf_reduced, train_category)

alphas = [x['for_cla__max_features'] for x in estimator_forcla.cv_results_['params']]
means = [x for x in estimator_forcla.cv_results_['mean_test_score']]
stds = [x for x in estimator_forcla.cv_results_['std_test_score']]
plt.figure(figsize=(8, 6))
plt.errorbar(alphas, means, stds, fmt='o', lw=1)
plt.plot(alphas, means)
plt.xlabel('max features')
plt.ylabel('mean cross validation R2')
plt.title('Random Forest')
plt.show()

print('\nBest number of features --->   ',estimator_forcla.best_params_['for_cla__max_features'])

cla_for_pipe1.set_params(for_cla__max_features = estimator_forcla.best_params_['for_cla__max_features'])
cla_for_pipe1.fit(corpus_train_tfidf_reduced, train_category)
R2_d=cla_for_pipe1.score(corpus_test_tfidf_reduced, test_category)
print ('\nRandom Forest Test score --->   ', R2_d)
print ('\nFeature importances:\n')
importances = cla_for_pipe1.named_steps['for_cla'].feature_importances_
indices = np.argsort(importances)[::-1]
for i in indices:
    print('%.3f' %importances[i],' <---  feature ',i)
# Plot the feature importances
plt.figure(figsize=(8, 6))
plt.title("Feature importances")
sns.barplot(indices, y=importances[indices])
plt.show()


estimators = []
estimators.append(('Normalizar', preprocessing.Normalizer()))
estimators.append(('boost_cla', AdaBoostClassifier()))
cla_boost_pipe1 = Pipeline(estimators)
cla_boost_pipe1.set_params(boost_cla__base_estimator=tree.DecisionTreeClassifier(max_depth=1), boost_cla__n_estimators=100, boost_cla__random_state=seed)

parameters = {
        'boost_cla__learning_rate' : np.logspace(-8,0,15)
}
estimator_boostcla = GridSearchCV(cla_boost_pipe1, parameters, cv=kfold)
                  
# evaluate the grid search and print best classifier
estimator_boostcla.fit(corpus_train_tfidf_reduced, train_category)

alphas = [x['boost_cla__learning_rate'] for x in estimator_boostcla.cv_results_['params']]
means = [x for x in estimator_boostcla.cv_results_['mean_test_score']]
stds = [x for x in estimator_boostcla.cv_results_['std_test_score']]
plt.figure(figsize=(8, 6))
plt.errorbar(alphas, means, stds, fmt='o', lw=1)
plt.plot(alphas, means)
plt.xscale('log')
plt.xlabel('learning rate')
plt.ylabel('mean cross validation R2')
plt.title('Adaptive Boosting')
plt.show()

print('\nBest learning rate --->   ',estimator_boostcla.best_params_['boost_cla__learning_rate'])

cla_boost_pipe1.set_params(boost_cla__learning_rate = estimator_boostcla.best_params_['boost_cla__learning_rate'])
cla_boost_pipe1.fit(corpus_train_tfidf_reduced, train_category)
R2_e=cla_boost_pipe1.score(corpus_train_tfidf_reduced, train_category)
print ('\nAdaptive boosting Test score --->   ', R2_e)






>>>>>>> 8981204fc2d5d2d4d6837fe1df4d45930a33d3af
