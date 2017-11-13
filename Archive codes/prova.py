import pandas as pd

seed = 123

df_complete = pd.read_json(path_or_buf = 'data/amazon_step1.json', lines=True)
df = pd.DataFrame(df_complete[['asin', 'category','reviewText']])
df = df.ix[:10000,:]
del df_complete

# Group comments by product and category assigned
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

# All the special characters were removed from the sample
df_refined=[]
for item in df.ix[:,2]:
    df_refined.append (item.replace('\r',' ').replace('/n',' ').replace('.',' ')\
                           .replace(',',' ').replace('(',' ').replace(')',' ')\
                           .replace("'s",' ').replace('"',' ').replace('!',' ')\
                           .replace('?',' ').replace("'",' ').replace('>',' ')\
                           .replace('$',' ').replace('-',' ').replace(';',' ')\
                           .replace(':',' ').replace('/',' ').replace('#',' '))
    

from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS

documents = df_refined

# remove the stopwords
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
from nltk.stem import WordNetLemmatizer
from nltk import download

download('wordnet')

lemmatizer = WordNetLemmatizer()

documents_no_stop_no_unique_no_numeric_lemmatize = [[lemmatizer.lemmatize(token) for token in text] 
                                                    for text in documents_no_stop_no_unique_no_numeric]

# remove non-english words
import enchant
eng_dic = enchant.Dict("en_US")

documents_no_stop_no_unique_no_numeric_lemmatize_english = [[token for token in text if (eng_dic.check(token)) ] 
                                                            for text in documents_no_stop_no_unique_no_numeric_lemmatize]

del documents_no_stop_no_unique_no_numeric_lemmatize

# create ready corpus
ready_corpus = documents_no_stop_no_unique_no_numeric_lemmatize_english

# build the dictionary and store it to disc for future use
dictionary = corpora.Dictionary(ready_corpus)

# convert the corpus into bag of words 
from gensim import models, corpora, matutils
dictionary = corpora.Dictionary(ready_corpus)
print(dictionary)

corpus_bow = [dictionary.doc2bow(comment) for comment in ready_corpus]

tfidf_transformer = models.TfidfModel(corpus_bow, normalize=True)

# apply tfidf transformation to the bow corpus
corpus_tfidf = tfidf_transformer [corpus_bow]

# convert to a sparse and compatible format for dimensionality reduction using sklearn
sparse_corpus_tfidf = matutils.corpus2csc(corpus_tfidf)
sparse_corpus_tfidf_transpose = sparse_corpus_tfidf.transpose()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sparse_corpus_tfidf_transpose, df.ix[:,1], test_size = 0.2, random_state = seed)


from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# reduce dimensions
print('Starting dimensionality reduction')
reducer= KernelPCA(n_components = 1500, kernel="cosine", random_state=seed)
corpus_train_tfidf_kpca = reducer.fit_transform(X_train)
corpus_test_tfidf_kpca = reducer.transform(X_test)

print('Finished dimensionality reduction')



#Initialize Logistic Regression
log_reg = LogisticRegression(C=1.0)
log_reg.fit(corpus_train_tfidf_kpca,y_train)

a=log_reg.score(corpus_test_tfidf_kpca,y_test)

print('Starting logistic regression 2')
log_reg.fit(X_train,y_train)

b=log_reg.score(X_test,y_test)



