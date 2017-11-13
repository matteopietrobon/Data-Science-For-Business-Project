import pandas as pd

seed = 123

df_complete = pd.read_json(path_or_buf = 'data/reviews_Digital_Music_5.json', lines=True)
df = pd.DataFrame(df_complete[['asin', 'overall','reviewText']])
df = df.ix[:5000,:]
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

print('Starting dimensionality reduction')

# reduce dimensions
from sklearn.decomposition import KernelPCA

reducer= KernelPCA(n_components = 200, kernel="cosine", random_state=seed)
corpus_train_tfidf_kpca = reducer.fit_transform(train_tfidf)
corpus_test_tfidf_kpca = reducer.transform(test_tfidf)

print('Finished dimensionality reduction')

#%%
from sklearn.dummy import DummyClassifier

X_train = corpus_train_tfidf_kpca 
X_test = corpus_test_tfidf_kpca
y_train = train_category
y_test = test_category

dummy_clf = DummyClassifier(strategy='stratified', random_state = seed)
dummy_clf.fit(X_train, y_train)
accuracy_dummy=dummy_clf.score(X_test,y_test)

print('Dummy Classifier Test Performance:', accuracy_dummy)

#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


X_train = corpus_train_tfidf_kpca 
X_test = corpus_test_tfidf_kpca
y_train = train_category
y_test = test_category

#Initialize K-Fold for cross validation
K = 5
kfold = KFold(n_splits=K, random_state=seed)

#Create Pipeline
estimators = []
estimators.append(('Normalizer', Normalizer()))
estimators.append(('knn_clf', KNeighborsClassifier()))
reg_knn_pipe1 = Pipeline(estimators)
reg_knn_pipe1.set_params(knn_clf__algorithm='ball_tree',knn_clf__weights='uniform')

#Create a grid search over n_neighbors values
parameters = {
        'knn_clf__n_neighbors' : np.arange(5,50)
}
estimator_knnreg = GridSearchCV(reg_knn_pipe1, parameters, cv=kfold)
                  
#Evaluate the grid search and print best regressor
print('Starting Grid Search')
estimator_knnreg.fit(X_train, y_train)

alphas = [x['knn_clf__n_neighbors'] for x in estimator_knnreg.cv_results_['params']]
means = [x for x in estimator_knnreg.cv_results_['mean_test_score']]
stds = [x for x in estimator_knnreg.cv_results_['std_test_score']]

plt.figure(figsize=(8, 6))
plt.errorbar(alphas, means, stds, fmt='o', lw=1)
plt.plot(alphas, means)
plt.xlabel('Neighbors')
plt.ylabel('Mean cross validation accuracy')
plt.title('KNN')
plt.show()

reg_knn_pipe1.set_params(knn_clf__n_neighbors = estimator_knnreg.best_params_['knn_clf__n_neighbors'])
reg_knn_pipe1.fit(X_train, y_train)

print("Best Number of Neighbors:", estimator_knnreg.best_params_['knn_clf__n_neighbors'])
accuracy_KNN=reg_knn_pipe1.score(X_test,y_test)
print ('\nTest score for Optimized KNN:', accuracy_KNN)

#%%

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
import seaborn as sns



K = 5
kfold = KFold(n_splits=K, random_state=seed)


n_tried=35
depths=np.arange(1,n_tried)

estimators = []
estimators.append(('Normalizer', Normalizer()))
estimators.append(('tree_cla', tree.DecisionTreeClassifier(random_state=seed)))
cla_tree_pipe1 = Pipeline(estimators)

parameters = {
        'tree_cla__max_depth' : depths
}
estimator_treecla = GridSearchCV(cla_tree_pipe1, parameters, cv=kfold)
                  
# evaluate the grid search and print best classifier
estimator_treecla.fit(X_train, y_train)

alphas = [x['tree_cla__max_depth'] for x in estimator_treecla.cv_results_['params']]
means = [x for x in estimator_treecla.cv_results_['mean_test_score']]
stds = [x for x in estimator_treecla.cv_results_['std_test_score']]

plt.figure(figsize=(8, 6))
plt.errorbar(alphas, means, stds, fmt='o', lw=1)
plt.plot(alphas, means)
plt.xlabel('max depth')
plt.ylabel('mean cross-validated Accuracy')
plt.title('Decision Tree')
plt.show()

print('\nBest max depth --->   ',estimator_treecla.best_params_['tree_cla__max_depth'])

cla_tree_pipe1.set_params(tree_cla__max_depth = estimator_treecla.best_params_['tree_cla__max_depth'])
cla_tree_pipe1.fit(X_train, y_train)
accuracy_tree = cla_tree_pipe1.score(X_test, y_test)
print ('\nTest score --->   ', accuracy_tree)
print ('\nFeature importances:\n')
importances = cla_tree_pipe1.named_steps['tree_cla'].feature_importances_
indices = np.argsort(importances)[::-1]


# Plot the feature importances
plt.figure(figsize=(8, 6))
plt.title("Feature importances")
sns.barplot(indices, y=importances[indices])
plt.show()


estimators = []
estimators.append(('Normalizer', Normalizer()))
estimators.append(('bag_cla', BaggingClassifier()))
cla_bag_pipe1 = Pipeline(estimators)
cla_bag_pipe1.set_params(bag_cla__base_estimator=tree.DecisionTreeClassifier(max_depth=estimator_treecla.best_params_['tree_cla__max_depth']),\
                         bag_cla__n_estimators=500, bag_cla__random_state=seed)

cla_bag_pipe1.fit(X_train,y_train)
accuracy_bagging=cla_bag_pipe1.score(X_test, y_test)
print ('\nBagging Test score --->   ', accuracy_bagging)

#%%

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
                  


K = 5
kfold = KFold(n_splits=K, random_state=seed)

# LINEAR KERNEL
estimators = []
estimators.append(('normalizer', Normalizer()))
estimators.append(('svm_linear_clf', SVC()))
svm_linear_pipe1 = Pipeline(estimators)
svm_linear_pipe1.set_params(svm_linear_clf__kernel='linear', svm_linear_clf__gamma='auto')

penalties = np.logspace(-5,3,9)


parameters = {
        'svm_linear_clf__C' : penalties
}
estimator_svm_linear = GridSearchCV(svm_linear_pipe1, parameters, cv=kfold)
                  
# evaluate the grid search and print best classifier
estimator_svm_linear.fit(X_train,y_train)

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

print('\nBest penalty for linear kernel --->   ',estimator_svm_linear.best_params_['svm_linear_clf__C'])

svm_linear_pipe1.set_params(svm_linear_clf__C = estimator_svm_linear.best_params_['svm_linear_clf__C'])
svm_linear_pipe1.fit(X_train,y_train)
accuracy_linearSVM = svm_linear_pipe1.score(X_test,y_test)
print ('\nLinear kernel test score --->   ', accuracy_linearSVM)



# GAUSSIAN KERNEL
estimators = []
estimators.append(('normalizer', Normalizer()))
estimators.append(('svm_gaussian_clf', SVC()))
svm_gaussian_pipe1 = Pipeline(estimators)
#No gridsearch on gamma, since 'auto' gave best results
svm_gaussian_pipe1.set_params(svm_gaussian_clf__kernel='rbf', svm_gaussian_clf__gamma='auto')

penalties = np.logspace(-5,3,9)


parameters = {
        'svm_gaussian_clf__C' : penalties
}
estimator_svm_gaussian = GridSearchCV(svm_gaussian_pipe1, parameters, cv=kfold)
                  
# evaluate the grid search and print best classifier
estimator_svm_gaussian.fit(X_train,y_train)

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

print('\nBest penalty for Gaussian kernel --->   ',estimator_svm_gaussian.best_params_['svm_gaussian_clf__C'])

svm_gaussian_pipe1.set_params(svm_gaussian_clf__C = estimator_svm_gaussian.best_params_['svm_gaussian_clf__C'])
svm_gaussian_pipe1.fit(X_train,y_train)
accuracy_gaussianSVM = svm_gaussian_pipe1.score(X_test,y_test)
print ('\nTest score --->   ', accuracy_gaussianSVM)



# LOGISTIC REGRESSION
estimators = []
estimators.append(('normalizer', Normalizer()))
estimators.append(('log_reg', LogisticRegression()))
log_reg_pipe1 = Pipeline(estimators)
log_reg_pipe1.set_params()

penalties = np.logspace(-5,3,9)


parameters = {
        'log_reg__C' : penalties
}
estimator_log_reg = GridSearchCV(log_reg_pipe1, parameters, cv=kfold)
                  
# evaluate the grid search and print best classifier
estimator_log_reg.fit(X_train,y_train)

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

print('\nBest penalty for Logistic Regression --->   ',estimator_log_reg.best_params_['log_reg__C'])

log_reg_pipe1.set_params(log_reg__C = estimator_log_reg.best_params_['log_reg__C'])
log_reg_pipe1.fit(X_train,y_train)
accuracy_logreg = log_reg_pipe1.score(X_test,y_test)
print ('\nLogistic regression test score --->   ', accuracy_logreg)






