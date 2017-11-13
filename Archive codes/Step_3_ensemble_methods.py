# -*- coding: utf-8 -*-
"""
Created on Sat May 20 13:58:42 2017

@author: Teo
"""


import pandas as pd

seed = 123

df_complete = pd.read_json(path_or_buf = 'data/reviews_Digital_Music_5.json', lines=True)
df = pd.DataFrame(df_complete[['asin', 'overall','reviewText']])
df = df.ix[:8000,:]
five_stars = df['overall']


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
scores=[analyzer.polarity_scores(sentence) for sentence in df.ix[:,2]]

import numpy as np
scores_array=np.array([(comment['neg'],comment['neu'],comment['pos']) for comment in scores])

#%%

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

X_validation, X_test, y_validation, y_test = train_test_split(scores_array,df.ix[:,1], test_size = 0.2, random_state = seed)

under_sampler=RandomUnderSampler(ratio=0.1,random_state=seed)
X_train, y_train = under_sampler.fit_sample(X_validation, y_validation)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(scores_array,df.ix[:,1], test_size = 0.2, random_state = seed)



from sklearn.metrics import confusion_matrix

def our_scoring_function(y_true,y_pred):
    penalties=np.zeros([5,5])
    for i in np.arange(5):
        for j in np.arange(5):
            penalties[i,j]= -np.abs(i-j)
    conf_mat = confusion_matrix(y_true, y_pred,labels=np.arange(1,6))/np.shape(y_true)[0]
    return np.sum(penalties*conf_mat)
    

from sklearn.metrics import make_scorer
our_scoring= make_scorer(our_scoring_function)

#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
import seaborn as sns



K = 5
kfold = KFold(n_splits=K, random_state=seed)


n_tried=15
depths=np.arange(1,n_tried)

estimators = []
estimators.append(('Normalizer', Normalizer()))
estimators.append(('tree_cla', tree.DecisionTreeClassifier(random_state=seed)))
cla_tree_pipe1 = Pipeline(estimators)

parameters = {
        'tree_cla__max_depth' : depths
}
estimator_treecla = GridSearchCV(cla_tree_pipe1, parameters,scoring=our_scoring, cv=kfold)
                  
# evaluate the grid search and print best classifier
estimator_treecla.fit(X_train, y_train)

alphas = [x['tree_cla__max_depth'] for x in estimator_treecla.cv_results_['params']]
means = [x for x in estimator_treecla.cv_results_['mean_test_score']]
stds = [x for x in estimator_treecla.cv_results_['std_test_score']]

plt.figure(figsize=(8, 6))
plt.errorbar(alphas, means, stds, fmt='o', lw=1)
plt.plot(alphas, means)
plt.xlabel('max depth')
plt.ylabel('mean cross-validated score')
plt.title('Decision Tree')
plt.show()

print('\nBest max depth --->   ',estimator_treecla.best_params_['tree_cla__max_depth'])

cla_tree_pipe1.set_params(tree_cla__max_depth = estimator_treecla.best_params_['tree_cla__max_depth'])
cla_tree_pipe1.fit(X_train, y_train)
score_tree=our_scoring_function(y_test,cla_tree_pipe1.predict(X_test))
print ('\nTest score for basic decision tree --->   ', score_tree)
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
score_bagging=our_scoring_function(y_test,cla_bag_pipe1.predict(X_test))
print ('\nBagging Test score --->   ', score_bagging)

#%%

from sklearn.linear_model import LogisticRegression
# LOGISTIC REGRESSION
estimators = []
estimators.append(('normalizer', Normalizer()))
estimators.append(('log_reg', LogisticRegression()))
log_reg_pipe1 = Pipeline(estimators)
log_reg_pipe1.set_params()

penalties = np.logspace(-5,5,7)


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


#%%
from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[('lr', log_reg_pipe1), ('bag', cla_bag_pipe1)], voting='hard')
eclf2 = VotingClassifier(estimators=[('lr', log_reg_pipe1), ('bag', cla_bag_pipe1)], voting='soft')
eclf1.fit(X_train,y_train)
score_eclf1=our_scoring_function(y_test,eclf1.predict(X_test))
print ('\nVoting classifier hard test score --->   ', score_eclf1)
eclf2.fit(X_train,y_train)
score_eclf2=our_scoring_function(y_test,eclf2.predict(X_test))
print ('\nVoting classifier soft test score --->   ', score_eclf2)



#%%
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

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")



y_cap = eclf2.predict(X_test)

Labels=np.unique(y_train)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_cap, labels=Labels)

# Plot confusion matrix
#plt.figure(2)
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

plot_confusion_matrix(cnf_matrix,Labels, normalize=False,
                      title='Confusion matrix\n')

plt.show()

