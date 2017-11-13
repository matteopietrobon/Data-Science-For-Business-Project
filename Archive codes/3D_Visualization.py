import pandas as pd

seed = 123

df_complete = pd.read_json(path_or_buf = 'data/amazon_step1.json', lines=True)
df = pd.DataFrame(df_complete[['asin', 'category','reviewText']])
df = df.ix[:5000,:]
del df_complete


united = df.groupby(['asin', 'category'])['reviewText'].apply(' '.join).reset_index()

old = 0

product_numbers = []

# create the list of the products with more than one category assigned
for _,row in united.iterrows():
    
    if(old == row['asin']):
        
        product_numbers.append(row['asin'])
        
    old = row['asin']

    
indexes=[]

# find what are the comments related to the products found above
for i,df_row in df.iterrows():
    
    if df_row['asin'] in product_numbers:
        
        indexes.append(i)

# drop the ambiguous observations
df.drop(df.index[[indexes]], inplace = True)

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

#remove non-english words
#import enchant
#eng_dic = enchant.Dict("en_US")
#
#documents_no_stop_no_unique_no_numeric_lemmatize_english = [[token for token in text if (eng_dic.check(token)) ] 
#                                                            for text in documents_no_stop_no_unique_no_numeric_lemmatize]

#print ('no english: ',documents_no_stop_no_unique_no_numeric_lemmatize_english[tester], '\n')
#del documents_no_stop_no_unique_no_numeric_lemmatize


#%%

# create ready corpus
#ready_corpus = documents_no_stop_no_unique_no_numeric_lemmatize_english
ready_corpus = documents_no_stop_no_unique_no_numeric_lemmatize
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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

kpca2 = KernelPCA(n_components = 3, kernel="cosine", random_state=seed)
corpus_train_tfidf_kpca2 = kpca2.fit_transform(sparse_train_corpus_tfidf_transpose)
#RENAMED FOR EASE
X = corpus_train_tfidf_kpca2

#kpca = KernelPCA(n_components = 1000 , kernel="cosine", random_state=seed)
#corpus_train_tfidf_kpca = kpca.fit_transform(sparse_train_corpus_tfidf_transpose)

#reducer = TSNE(n_components = 3, perplexity=.0, early_exaggeration=4.0, learning_rate=30.0, n_iter=10000, metric='cosine')
# visualize the tf-idf corpus using kernel PCA

#CREATE DICTIONARY TO ASSIGN COLORS
categories = train_category.unique()

#REINDEX OUTPUT TO COMPARE WITH LABELS
reindexed = train_category.reset_index()

#CREATE DICTIONARY TO ASSIGN VALUES TO OCCURRENCE VECTOR AND COLORS
together = zip(categories,np.arange(0,24))
locator = dict(together)
occurred =  np.zeros(24)

reds = [1, 0.6, 0.86, 1, 0,0.4,  1,0.6,0.5,  1,0.5,  0,0.25,0.2,  1,1,0.6,  0,0.5,0.75,0.8,0.6,1,0.13]
greens =[0,0.3, 0.67,0.8,0,0.8,0.5,0.2,0.5,  0,  1,0.5,0.25,  1,0.4,1,  1,0.8,0.5,0.75,0.4,0.3,0,0.13]
blues = [0, 0, 0.3, 1,   1,  1,  0,  1,0.5,0.5,  1,  1,0.25,0.2,  1,0,0.9,  0,0.5,0.75,  0,  0,1,0.13]   

markers = ['o', 'v', '+', '<', '>', 'h', 'd', '^', 'x', 'o', 's', 'p', 'd', '+', '+', 'x', 'd','o', 'v', '^', '<', '>', 'o', '+' ]
colori = pd.DataFrame(reds, columns = ['Reds'])
colori['Greens'] = greens
colori['Blues'] = blues

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(1,len(train_category)):

    index = locator[reindexed.ix[i,1]]
    
    if(occurred[locator[reindexed.ix[i,1]]] == 0):
        
    
        occurred[locator[reindexed.ix[i,1]]] = 1
        ax.scatter(X[i,0], X[i,1], X[i,2], color = colori.ix[index,:], label = reindexed.ix[i,1], marker=markers[index])
        
    else:
        #NO LABEL IF ALREADY EXISTING
        ax.scatter(X[i,0], X[i,1], X[i,2], color = colori.ix[index,:],marker=markers[index])
        
plt.legend()
plt.title("3D Data Projection")
