#
import pandas as pd

seed = 123

df_complete = pd.read_json(path_or_buf = 'data/reviews_Digital_Music_5.json', lines=True)
df = pd.DataFrame(df_complete[['asin', 'overall','reviewText']])
df = df.ix[:2000,:]
del df_complete

stars= pd.get_dummies(df['overall'])
print(stars.sum())


#five_stars = (df['overall']==5)*1.0
five_stars = df['overall']

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
#import enchant
#eng_dic = enchant.Dict("en_US")
#
## remove non-english words
#documents_no_stop_no_unique_no_numeric_lemmatize_english = [[token for token in text if (eng_dic.check(token)) ] 
#                                                            for text in documents_no_stop_no_unique_no_numeric_lemmatize]
#
#print ('no english: ',documents_no_stop_no_unique_no_numeric_lemmatize_english[tester], '\n')
#del documents_no_stop_no_unique_no_numeric_lemmatize


#%%

# create ready corpus
#ready_corpus = documents_no_stop_no_unique_no_numeric_lemmatize_english
ready_corpus = documents_no_stop_no_unique_no_numeric_lemmatize
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

#%%
train_tfidf, test_tfidf, train_category, test_category = train_test_split(sparse_corpus_tfidf_transpose, five_stars, test_size = 0.2, random_state = seed)

print('starting dimensionality reduction')
# reduce dimensions
from sklearn.decomposition import KernelPCA
reducer= KernelPCA(n_components = 30 , kernel="cosine", random_state=seed)
corpus_train_tfidf_kpca = reducer.fit_transform(train_tfidf)
corpus_test_tfidf_kpca = reducer.transform(test_tfidf)

corpus_train_tfidf_reduced=corpus_train_tfidf_kpca
corpus_test_tfidf_reduced =corpus_test_tfidf_kpca

#%%

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
                  


K = 5
kfold = KFold(n_splits=K, random_state=seed)


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



#%%
# visualize the tf-idf corpus using kernel PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from mpl_toolkits.mplot3d import Axes3D

kpca = KernelPCA(n_components = 3, kernel="cosine", random_state=seed)
corpus_tfidf_kpca = kpca.fit_transform(sparse_corpus_tfidf_transpose)

#RENAMED FOR EASE
X = corpus_tfidf_kpca

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

fig = plt.figure()
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
