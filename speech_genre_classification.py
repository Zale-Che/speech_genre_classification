#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import random
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ## Data

# In[2]:


## load data
corpus = pd.read_csv('data/ML_corpus_v3.tsv', sep='\t', header=0)
corpus.head()


# In[3]:


## Check Genre Distribution of the Current Dataset
from collections import Counter
corpus_genre = corpus['genre']
genre_distr = Counter(corpus_genre)
print(genre_distr)


# In[4]:


## corpus for text only
corpus_txt = corpus['text']
corpus_txt.head()


# In[ ]:


## corpus for acoustic only
corpus_acou = corpus.drop(labels=["genre","speaker_ID", "text"], axis=1)
corpus_acou.head()


# In[ ]:


# split the data
from sklearn.model_selection import train_test_split
train, test = train_test_split(corpus, test_size = 0.2, random_state = 777)


# In[ ]:


print(len(train))
print(len(test))


# In[ ]:


# check dist. after splitting
train_genre_distr = Counter(train["genre"])
print(train_genre_distr)
test_genre_distr = Counter(test["genre"])
print(test_genre_distr)


# In[ ]:


# check dist. after splitting
train_genre_distr = Counter(train["speaker_ID"])
print(train_genre_distr)
test_genre_distr = Counter(test["speaker_ID"])
print(test_genre_distr)


# # Prosodic features

# In[ ]:


# x = feature; y = label
p_x_train = train.drop(labels=["genre", "speaker_ID", "text", "f0_reset"], axis=1) ## remove f0_reset
p_x_test = test.drop(labels=["genre", "speaker_ID", "text", "f0_reset"], axis=1)
p_y_train = train["genre"]
p_y_test = test["genre"]


# In[ ]:


# check prosodic features
p_features = list(p_x_train.columns)
p_features


# In[ ]:


# scaling
sc = StandardScaler()
p_x_train = sc.fit_transform(p_x_train)
p_x_test = sc.transform(p_x_test)


# ## SVM

# In[ ]:


# fit the model
from sklearn import svm

p_model_svm = svm.SVC(C=1.0, kernel='rbf')
p_model_svm.fit(p_x_train, p_y_train)


# In[ ]:


# cross validation
from sklearn.model_selection import cross_val_score
p_model_svm_acc = cross_val_score(estimator=p_model_svm, X=p_x_train, y=p_y_train, cv=5, n_jobs=-1) ## cv=5: 20% of the data is used for testing
print(p_model_svm_acc)
print(np.mean(p_model_svm_acc))
print(np.std(p_model_svm_acc))


# In[ ]:


# plotting cross validation accuracy
fold = [1, 2, 3, 4, 5]
plt.ylim(0.6, 0.75)
plt.bar(fold, p_model_svm_acc)
plt.xlabel('kth fold')
plt.ylabel('Accuracy')
plt.show()


# ## Random forests

# In[ ]:


# fit the model
from sklearn.ensemble import RandomForestClassifier
p_model_forest = RandomForestClassifier(n_estimators=100, # the number of trees; the larger the better, but compute longer
                                       max_features="sqrt", # "sqrt" / None
                                       criterion="entropy",
                                       random_state=0) # the size of the random subsets of features to consider when splitting a node; try None; 1.0; sqrt
p_model_forest.fit(p_x_train, p_y_train)


# In[ ]:


# cross validation
p_model_forest_acc = cross_val_score(estimator=p_model_forest, X=p_x_train, y=p_y_train, cv=5, n_jobs=-1) ## cv=5: 20% of the data is used for testing
print(p_model_forest_acc)
print(np.mean(p_model_forest_acc))
print(np.std(p_model_forest_acc))


# In[ ]:


# plotting cross validation accuracy
fold = [1, 2, 3, 4, 5]
plt.ylim(0.6, 0.75)
plt.bar(fold, p_model_forest_acc)
plt.xlabel('kth fold')
plt.ylabel('Accuracy')
plt.show()


# ## Accuracy

# In[ ]:


print(p_model_svm.score(p_x_test, p_y_test))  ## SVM
print(p_model_forest.score(p_x_test, p_y_test)) ## random forests


# ## Precision

# In[ ]:


from sklearn.metrics import precision_score
y_pred_p_model_svm = p_model_svm.predict(p_x_test)
precision_score(p_y_test, y_pred_p_model_svm,  # svm
         labels = None)


# In[ ]:


from sklearn.metrics import precision_score
y_pred_p_model_forest = p_model_forest.predict(p_x_test)
precision_score(p_y_test, y_pred_p_model_forest,  # random forests
         labels = None)


# ## Recall

# In[ ]:


from sklearn.metrics import recall_score
y_pred_p_model_svm = p_model_svm.predict(p_x_test)
recall_score(p_y_test, y_pred_p_model_svm,  # svm
         labels = None)


# In[ ]:


from sklearn.metrics import recall_score
y_pred_p_model_forest = p_model_forest.predict(p_x_test)
recall_score(p_y_test, y_pred_p_model_forest,  # random forests
         labels = None)


# ## F1 score

# In[ ]:


from sklearn.metrics import f1_score
y_pred_p_model_svm = p_model_svm.predict(p_x_test)
f1_score(p_y_test, y_pred_p_model_svm,  # svm
         labels = None)


# In[ ]:


from sklearn.metrics import f1_score
y_pred_p_model_forest = p_model_forest.predict(p_x_test)
f1_score(p_y_test, y_pred_p_model_forest, # random forests
         labels = None)


# ## Confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cfm = confusion_matrix(p_y_test, y_pred_p_model_svm, normalize="all") # svm
dis = ConfusionMatrixDisplay(cfm)
dis.plot()
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cfm = confusion_matrix(p_y_test, y_pred_p_model_forest, normalize="all") # random forests
dis = ConfusionMatrixDisplay(cfm)
dis.plot()
plt.show()


# In[ ]:


#%%time
# tuning Model Hyperparameters (SVM)
#from sklearn.model_selection import GridSearchCV

#parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': (1,4,8,16,32), 'gamma': ('scale', 'auto')}

#svc = svm.SVC()
#clf = GridSearchCV(svc, parameters, cv=5, n_jobs=-1) ## `-1` run in parallel
#clf.fit(p_x_train, p_y_train)

#print(clf.best_params_) # {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
#print(clf.score(p_x_test, p_y_test)) # 0.7188118811881188


# In[ ]:


#%%time
# tuning Model Hyperparameters (random forests)
#from sklearn.model_selection import GridSearchCV

#parameters = {'n_estimators': (100, 300, 500),'criterion': ('gini', 'entropy'), 'max_features': ('sqrt', None)}

#rfc = RandomForestClassifier(random_state = 0)
#clf = GridSearchCV(rfc, parameters, cv=5, n_jobs=-1) ## `-1` run in parallel
#clf.fit(p_x_train, p_y_train)

#print(clf.best_params_) # {'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 100}
#print(clf.score(p_x_test, p_y_test)) # 0.7326732673267327


# ## Feature importance

# In[ ]:


from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
perm_importance = permutation_importance(p_model_svm, p_x_test, p_y_test, random_state=0)

feature_names = ['utterance_duration', 'pause_duration', 'duration_PVI', 'speech_rate', 'f0_mean', 'f0_range', 'f0_PVI']
#feature_names = ['utterance_duration', 'pause_duration', 'duration_PVI', 'speech_rate', 'f0_mean', 'f0_range', 'f0_PVI', 'f0_reset']
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")


# In[ ]:


# visualize a tree in random forests
import graphviz
from sklearn.tree import export_graphviz

p_fig_data = export_graphviz(p_model_forest.estimators_[99], 
                           feature_names=p_features,
                           class_names=["0", "1"], 
                           filled=True, impurity=True, 
                           rounded=True)

p_fig = graphviz.Source(p_fig_data, format='png')
#p_fig


# In[ ]:


# save the plot
#graph.render('p_fig')


# # Text features

# In[ ]:


# x = feature; y = label
t_x_train = train["text"]
t_x_test = test["text"]
t_y_train = train["genre"]
t_y_test = test["genre"]


# In[ ]:


# check data
t_x_train = np.array(t_x_train)
t_y_train = np.array(t_y_train)
print(t_x_train)
print(t_y_train)
t_x_test = np.array(t_x_test)
t_y_test = np.array(t_y_test)
print(t_x_test)
print(t_y_test)


# In[ ]:


# preprocessing
def normalize_document(doc):
    # remove special characters\whitespaces
    doc = re.sub(r'[\s]', '', doc)
    doc = doc.strip()
    return doc

normalize_corpus = np.vectorize(normalize_document)  # 如果一個python函數只能對單個元素進行某種處理操作，經過vectorize轉化之後，能夠實現對一個數組進行處理


# In[ ]:


# remove redundant whitespaces
t_x_train = normalize_corpus(t_x_train)
t_x_test = normalize_corpus(t_x_test)


# In[ ]:


# tokenize
ws_driver = CkipWordSegmenter(device=-1)
tokens_train = ws_driver(t_x_train, use_delim=True)
tokens_test = ws_driver(t_x_test, use_delim=True)


# In[ ]:


tokens_train[:6]


# In[ ]:


# concatenate tokens back to utterances
utts_train = [" ".join(t) for t in tokens_train]
utts_train = np.array(utts_train)
utts_test = [" ".join(t) for t in tokens_test]
utts_test = np.array(utts_test)


# In[ ]:


print(utts_train)
print(utts_test)


# ## text vectorization

# In[ ]:


## text vectorization: TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(min_df=2, ## cut-off; ignore terms that have a document frequency strictly lower than the given threshold.
                     max_df=1.0,
                     norm='l2', ## notice: norm='l2' l is a letter instead of number 1
                     use_idf=True,
                     smooth_idf=True,
                     ngram_range=(1, 1)) # only unigrams; (1, 2) means unigrams and bigrams
x_train_bow = tv.fit_transform(utts_train)  ## .fit_transform: put into training data; .transform: put into testing data
x_test_bow = tv.transform(utts_test) # transform test


## higher number: semantically more important 
# The L1 norm will drive some weights to 0, inducing sparsity in the weights. This can be beneficial for memory efficiency or when feature selection is needed (i.e., we want to select only certain weights).
# The L2 norm instead will reduce all weights but not all the way to 0. This is less memory efficient but can be useful if we want/need to retain all parameters.


# In[ ]:


print(x_train_bow.shape)
print(x_test_bow.shape)


# In[ ]:


# get all unique words in the corpus
vocab = tv.get_feature_names_out()
vocab


# In[ ]:


# TF-IDF score
# sum tfidf frequency of each term through documents
sums = x_train_bow.sum(axis=0)

# connecting term to its sums frequency
data = []
for col, term in enumerate(vocab):
    data.append( (term, sums[0,col] ))

ranking = pd.DataFrame(data, columns=['term','rank'])
print(ranking.sort_values('rank', ascending=False))


# In[ ]:


# top TF-IDF
top_n = 50
print('tf_idf scores: \n', sorted(list(zip(vocab, 
                                           x_train_bow.sum(0).getA1())), 
                                 key=lambda x: x[1], reverse=True)[:top_n])


# In[ ]:


# top IDF
top_n = 1586
print('idf values: \n', sorted(list(zip(vocab,tv.idf_,)),
       key = lambda x: x[1], reverse=True)[:top_n])


# ## SVM

# In[ ]:


# fit the model
from sklearn import svm

t_model_svm = svm.SVC(C=1.0, kernel='rbf')
t_model_svm.fit(x_train_bow, t_y_train)


# In[ ]:


# cross validation
from sklearn.model_selection import cross_val_score
t_model_svm_acc = cross_val_score(estimator=t_model_svm, X=x_train_bow, y=t_y_train, cv=5, n_jobs=-1) ## cv=5: 20% of the data is used for testing
print(t_model_svm_acc)
print(np.mean(t_model_svm_acc))
print(np.std(t_model_svm_acc))


# In[ ]:


# plotting cross validation accuracy
fold = [1, 2, 3, 4, 5]
plt.ylim(0.7, 0.85)
plt.bar(fold, t_model_svm_acc)
plt.xlabel('kth fold')
plt.ylabel('Accuracy')
plt.show()


# ## Random forests

# In[ ]:


# fit the model
from sklearn.ensemble import RandomForestClassifier
t_model_forest = RandomForestClassifier(n_estimators=500, # the number of trees; the larger the better, but compute longer
                                       max_features="sqrt", # "sqrt" is better than None
                                       criterion="gini",
                                       random_state=0) # the size of the random subsets of features to consider when splitting a node; try None; 1.0; sqrt
t_model_forest.fit(x_train_bow, t_y_train)


# In[ ]:


# cross validation
t_model_forest_acc = cross_val_score(estimator=t_model_forest, X=x_train_bow, y=t_y_train, cv=5, n_jobs=-1) ## cv=5: 20% of the data is used for testing
print(t_model_forest_acc)
print(np.mean(t_model_forest_acc))
print(np.std(t_model_forest_acc))


# In[ ]:


# plotting cross validation accuracy
fold = [1, 2, 3, 4, 5]
plt.ylim(0.7, 0.85)
plt.bar(fold, t_model_forest_acc)
plt.xlabel('kth fold')
plt.ylabel('Accuracy')
plt.show()


# ## Accuracy

# In[ ]:


print(t_model_svm.score(x_test_bow, t_y_test))  ## SVM
print(t_model_forest.score(x_test_bow, t_y_test)) ## random forests


# ## Precision

# In[ ]:


from sklearn.metrics import precision_score
y_pred_t_model_svm = t_model_svm.predict(x_test_bow)
precision_score(t_y_test, y_pred_t_model_svm, # svm
         labels = None)


# In[ ]:


from sklearn.metrics import precision_score
y_pred_t_model_forest = t_model_forest.predict(x_test_bow)
precision_score(t_y_test, y_pred_t_model_forest,  # random forest
         labels = None)


# ## Recall

# In[ ]:


from sklearn.metrics import recall_score
y_pred_t_model_svm = t_model_svm.predict(x_test_bow)
recall_score(t_y_test, y_pred_t_model_svm, # svm
         labels = None)


# In[ ]:


from sklearn.metrics import recall_score
y_pred_t_model_forest = t_model_forest.predict(x_test_bow)
recall_score(t_y_test, y_pred_t_model_forest,  # random forest
         labels = None)


# ## F1 score

# In[ ]:


from sklearn.metrics import f1_score
y_pred_t_model_svm = t_model_svm.predict(x_test_bow)
f1_score(t_y_test, y_pred_t_model_svm,  # svm
         labels = None)


# In[ ]:


from sklearn.metrics import f1_score
y_pred_t_model_forest = t_model_forest.predict(x_test_bow)
f1_score(t_y_test, y_pred_t_model_forest, # random forest
         labels = None)


# ## Confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cfm = confusion_matrix(t_y_test, y_pred_t_model_svm, normalize="all") # svm
dis = ConfusionMatrixDisplay(cfm)
dis.plot()
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cfm = confusion_matrix(t_y_test, y_pred_t_model_forest, normalize="all") # random forest
dis = ConfusionMatrixDisplay(cfm)
dis.plot()
plt.show()


# ## Feature importance

# In[ ]:


# time-consuming,uncomment if need a run
#from sklearn.inspection import permutation_importance
#import matplotlib.pyplot as plt
#%matplotlib inline
#x_test_bow_array = x_test_bow.toarray()
#perm_importance_t = permutation_importance(t_model_svm, x_test_bow_array, t_y_test, random_state=0)


# In[ ]:


# read permutation results
import pickle
with open('./t_permutation.pkl', 'rb') as f:
    perm_importance_t = pickle.load(f)

# find positive score
for i in perm_importance_t.importances_mean.argsort()[::-1]:
    if perm_importance_t.importances_mean[i] - 2 * perm_importance_t.importances_std[i] > 0: 
        print(f"{vocab[i]:<8}"
              f"{perm_importance_t.importances_mean[i]:.3f}"
              f" +/- {perm_importance_t.importances_std[i]:.3f}")


# In[ ]:


# read permutation results
import pickle
with open('./t_permutation.pkl', 'rb') as f:
    perm_importance_t = pickle.load(f)

# find negative score
for i in perm_importance_t.importances_mean.argsort():
    if perm_importance_t.importances_mean[i] - 2 * perm_importance_t.importances_std[i] < 0: 
        print(f"{vocab[i]:<8}"
              f"{perm_importance_t.importances_mean[i]:.3f}"
              f" +/- {perm_importance_t.importances_std[i]:.3f}")


# In[ ]:


# plotting
import matplotlib
matplotlib.rcParams['font.family'] = ['Heiti TC'] # for chinese label
feature_names_t = vocab
features_t = np.array(feature_names_t)
sorted_idx_t = perm_importance_t.importances_mean.argsort()
plt.barh(features_t[sorted_idx_t[::-1][:10][::-1]], perm_importance_t.importances_mean[sorted_idx_t[::-1][:10][::-1]])
plt.xlabel("Permutation Importance")


# In[ ]:


# store permutation results
#import pickle
#with open('../PyLibrary/t_permutation.pkl', 'wb') as f:
#    pickle.dump(perm_importance_t, f)


# In[ ]:


#%%time
# tuning Model Hyperparameters (SVM)
#from sklearn.model_selection import GridSearchCV

#parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': (1,4,8,16,32), 'gamma': ('scale', 'auto')}

#svc = svm.SVC()
#clf = GridSearchCV(svc, parameters, cv=5, n_jobs=-1) ## `-1` run in parallel
#clf.fit(x_train_bow, t_y_train)

#print(clf.best_params_) # {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
#print(clf.score(x_test_bow, t_y_test)) # 0.8267326732673267


# In[ ]:


#%%time
# tuning Model Hyperparameters (random forests)
#from sklearn.model_selection import GridSearchCV

#parameters = {'n_estimators': (100, 300, 500),'criterion': ('gini', 'entropy'), 'max_features': ('sqrt', None)}

#rfc = RandomForestClassifier(random_state = 0)
#clf = GridSearchCV(rfc, parameters, cv=5, n_jobs=-1) ## `-1` run in parallel
#clf.fit(x_train_bow, t_y_train)

#print(clf.best_params_) # {'criterion': 'gini', 'max_features': 'sqrt', 'n_estimators': 500}
#print(clf.score(x_test_bow, t_y_test)) # 0.7871287128712872


# # Text and prosodic features

# In[ ]:


## prosodic features
print(p_x_train.shape)
print(p_x_test.shape)
print(p_y_train.shape)
print(p_y_test.shape)


# In[ ]:


## text features
print(x_train_bow.shape)
print(x_test_bow.shape)
print(t_y_train.shape)
print(t_y_test.shape)


# In[ ]:


## create integrated features
int_x_train = np.concatenate((np.array(p_x_train), x_train_bow.toarray()), axis = 1)
int_x_test = np.concatenate((np.array(p_x_test), x_test_bow.toarray()), axis = 1)
print(int_x_train.shape)
print(int_x_test.shape)


# ## SVM

# In[ ]:


# fit the model
from sklearn import svm

int_model_svm = svm.SVC(C=1.0, kernel='linear')
int_model_svm.fit(int_x_train, t_y_train)


# In[ ]:


# corss validation
from sklearn.model_selection import cross_val_score
int_model_svm_acc = cross_val_score(estimator=int_model_svm, X=int_x_train, y=t_y_train, cv=5, n_jobs=-1) ## cv=5: 20% of the data is used for testing
print(int_model_svm_acc)
print(np.mean(int_model_svm_acc))
print(np.std(int_model_svm_acc))


# In[ ]:


# plotting cross validation accuracy
fold = [1, 2, 3, 4, 5]
plt.ylim(0.6, 0.85)
plt.bar(fold, int_model_svm_acc)
plt.xlabel('kth fold')
plt.ylabel('Accuracy')
plt.show()


# ## Random forests

# In[ ]:


# fit the model
from sklearn.ensemble import RandomForestClassifier
int_model_forest = RandomForestClassifier(n_estimators=500, # the number of trees; the larger the better, but compute longer
                                       max_features="sqrt", # "sqrt" is better than None
                                       criterion="entropy", # gini 表現較差
                                       random_state=0) # the size of the random subsets of features to consider when splitting a node; try None; 1.0; sqrt
int_model_forest.fit(int_x_train, t_y_train)


# In[ ]:


# corss validation
int_model_forest_acc = cross_val_score(estimator=int_model_forest, X=int_x_train, y=t_y_train, cv=5, n_jobs=-1) ## cv=5: 20% of the data is used for testing
print(int_model_forest_acc)
print(np.mean(int_model_forest_acc))
print(np.std(int_model_forest_acc))


# In[ ]:


# plotting cross validation accuracy
fold = [1, 2, 3, 4, 5]
plt.ylim(0.6, 0.85)
plt.bar(fold, int_model_forest_acc)
plt.xlabel('kth fold')
plt.ylabel('Accuracy')
plt.show()


# ## Accuracy

# In[ ]:


print(int_model_svm.score(int_x_test, t_y_test))  ## SVM
print(int_model_forest.score(int_x_test, t_y_test)) ## random forests


# ## Precision

# In[ ]:


from sklearn.metrics import precision_score
y_pred_int_model_svm = int_model_svm.predict(int_x_test) # svm
precision_score(t_y_test, y_pred_int_model_svm, 
         labels = None)


# In[ ]:


from sklearn.metrics import precision_score
y_pred_int_model_forest = int_model_forest.predict(int_x_test) # random forest
precision_score(t_y_test, y_pred_int_model_forest, 
         labels = None)


# ## Recall

# In[ ]:


from sklearn.metrics import recall_score
y_pred_int_model_svm = int_model_svm.predict(int_x_test) # svm
recall_score(t_y_test, y_pred_int_model_svm, 
         labels = None)


# In[ ]:


from sklearn.metrics import recall_score
y_pred_int_model_forest = int_model_forest.predict(int_x_test) # random forest
recall_score(t_y_test, y_pred_int_model_forest, 
         labels = None)


# ## F1 score

# In[ ]:


from sklearn.metrics import f1_score
y_pred_int_model_svm = int_model_svm.predict(int_x_test) # svm
f1_score(t_y_test, y_pred_int_model_svm, 
         labels = None)


# In[ ]:


from sklearn.metrics import f1_score
y_pred_int_model_forest = int_model_forest.predict(int_x_test) # random forest
f1_score(t_y_test, y_pred_int_model_forest, 
         labels = None)


# ## Confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cfm = confusion_matrix(t_y_test, y_pred_int_model_svm, normalize="all") # svm
dis = ConfusionMatrixDisplay(cfm)
dis.plot()
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cfm = confusion_matrix(t_y_test, y_pred_int_model_forest, normalize="all") # random forest
dis = ConfusionMatrixDisplay(cfm)
dis.plot()
plt.show()


# In[ ]:


#%%time
# tuning Model Hyperparameters (SVM)
#from sklearn.model_selection import GridSearchCV

#parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': (1,4,8,16,32), 'gamma': ('scale', 'auto')}

#svc = svm.SVC()
#clf = GridSearchCV(svc, parameters, cv=5, n_jobs=-1) ## `-1` run in parallel
#clf.fit(int_x_train, t_y_train)

#print(clf.best_params_) # {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}
#print(clf.score(int_x_test, t_y_test)) # 0.8366336633663366


# In[ ]:


#%%time
# tuning Model Hyperparameters (random forests)
#from sklearn.model_selection import GridSearchCV

#parameters = {'n_estimators': (100, 300, 500),'criterion': ('gini', 'entropy'), 'max_features': ('sqrt', None)}

#rfc = RandomForestClassifier(random_state = 0)
#clf = GridSearchCV(rfc, parameters, cv=5, n_jobs=-1) ## `-1` run in parallel
#clf.fit(int_x_train, t_y_train)

#print(clf.best_params_) # {'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 500}
#print(clf.score(int_x_test, t_y_test)) # 0.7920792079207921


# # Esemble model

# In[ ]:


from sklearn.datasets import make_classification
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


# In[ ]:


# integrated features
int2_x_train = np.concatenate((np.array(p_x_train), x_train_bow.toarray()), axis = 1)
int2_x_test = np.concatenate((np.array(p_x_test), x_test_bow.toarray()), axis = 1)
print(int2_x_train.shape)
print(int2_x_test.shape)


# In[ ]:


# label
int2_y_train = t_y_train
int2_y_test = t_y_test
print(int2_y_train.shape)
print(int2_y_test.shape)


# In[ ]:


# Base estimator (text):
clf_1 = svm.SVC(C=1.0, kernel='rbf')
clf_1_transformer = FunctionTransformer(lambda X: X[:, 7:])  ## select text columns without considering f0-reset
#clf_1_transformer = FunctionTransformer(lambda X: X[:, 8:])  ## select text columns with f0-reset considered
tclf_1 = Pipeline(
    [('transformer_1', clf_1_transformer), ('clf_1', clf_1)]
)

# Base estimator 2 (prosody):
clf_2 = RandomForestClassifier(n_estimators=100, # the number of trees; the larger the better, but compute longer
                                       max_features="sqrt", # "sqrt" is better than None
                                       criterion="entropy",
                                       random_state=0) # the size of the random subsets of features to consider when splitting a node; try None; 1.0; sqrt
clf_2_transformer = FunctionTransformer(lambda X: X[:, :7])  ## select prosody columns without considering f0-reset
#clf_2_transformer = FunctionTransformer(lambda X: X[:, :8])  ## select prosody columns with f0-reset considered
tclf_2 = Pipeline(
    [('transformer_2', clf_2_transformer), ('clf_2', clf_2)]
)


# In[ ]:


# The meta-learner (SVM) uses the transformed-classifiers as base estimators:
meta_model = svm.SVC(C=1.0, kernel='linear')

# Stacking: Training of the final estimator happens via cross-validation. 
esemble_model = StackingClassifier(estimators=[('tclf_1', tclf_1), ('tclf_2', tclf_2)], final_estimator = meta_model, n_jobs = -1, cv = 10) # cv here is not used for model evaluation but for prediction
esemble_model.fit(int2_x_train, int2_y_train)


# In[ ]:


# cross validation
esemble_model_acc = cross_val_score(estimator=esemble_model, X=int2_x_train, y=int2_y_train, cv=5, n_jobs=-1) ## cv=5: 20% of the data is used for testing
print(esemble_model_acc)
print(np.mean(esemble_model_acc))
print(np.std(esemble_model_acc))


# In[ ]:


# plotting cross validation accuracy
fold = [1, 2, 3, 4, 5]
plt.ylim(0.6, 0.87)
plt.bar(fold, esemble_model_acc)
plt.xlabel('kth fold')
plt.ylabel('Accuracy')
plt.show()


# ## Accuracy

# In[ ]:


print(esemble_model.score(int2_x_test, int2_y_test))


# ## Precision

# In[ ]:


y_pred_esemble_model = esemble_model.predict(int2_x_test)
precision_score(int2_y_test, y_pred_esemble_model, 
         labels = None)


# ## Recall

# In[ ]:


y_pred_esemble_model = esemble_model.predict(int2_x_test)
recall_score(int2_y_test, y_pred_esemble_model, 
         labels = None)


# ## F1 score

# In[ ]:


y_pred_esemble_model = esemble_model.predict(int2_x_test)
f1_score(int2_y_test, y_pred_esemble_model, 
         labels = None)


# ## Confusion matrix

# In[ ]:


cfm = confusion_matrix(int2_y_test, y_pred_esemble_model, normalize="all")
dis = ConfusionMatrixDisplay(cfm)
dis.plot()
plt.show()


# In[ ]:


#%%time
# tuning Model Hyperparameters (SVM as the meta-learner)
#from sklearn.model_selection import GridSearchCV
#parameters = {'final_estimator__kernel': ['linear', 'rbf', 'poly'], 'final_estimator__C': [1,4,8,16,32], 'final_estimator__gamma': ['scale', 'auto']}


#clf = GridSearchCV(esemble_model, parameters, cv=5, n_jobs=-1) ## `-1` run in parallel
#clf.fit(int2_x_train, int2_y_train)

#print(clf.best_params_) # {'final_estimator__C': 32, 'final_estimator__gamma': 'scale', 'final_estimator__kernel': 'rbf'}
#print(clf.score(int2_x_test, int2_y_test)) # 0.8455445544554455


# In[ ]:


#%%time
# If Random Forest as the meta-learner (worse performance)
#meta_model = RandomForestClassifier(random_state=0)

# Stacking: Training of the final estimator happens via cross-validation. 
#esemble_model = StackingClassifier(estimators=[('tclf_1', tclf_1), ('tclf_2', tclf_2)], final_estimator = meta_model, n_jobs = -1, cv = 10) # cv is not used for model evaluation but for prediction
#esemble_model.fit(int2_x_train, int2_y_train)

# tuning Model Hyperparameters (random forests)
#from sklearn.model_selection import GridSearchCV
#parameters = {'final_estimator__n_estimators': [100, 300, 500],'final_estimator__criterion': ['gini', 'entropy'], 'final_estimator__max_features': ['sqrt', None]}


#clf = GridSearchCV(esemble_model, parameters, cv=5, n_jobs=-1) ## `-1` run in parallel
#clf.fit(int2_x_train, int2_y_train)

#print(clf.best_params_) # {'final_estimator__criterion': 'entropy', 'final_estimator__max_features': None, 'final_estimator__n_estimators': 300}
#print(clf.score(int2_x_test, int2_y_test)) # 0.804950495049505


# ## Classifier importance

# In[ ]:


# prosody classifier
esem_p_cl = esemble_model.estimators_[1]
get_ipython().run_line_magic('matplotlib', 'inline')
perm_importance = permutation_importance(esem_p_cl, int2_x_test, int2_y_test, random_state=0) ## importances_mean長度為1593(prosody 7 + text 1586)，前面7個features為prosodic features
p_importance = perm_importance.importances_mean[0:7]  ## 只需要前面7個prosodic features的數值
#p_importance = perm_importance.importances_mean[0:8]  ## 只需要前面8個prosodic features的數值 (considering f0-reset)

# plotting
sorted_idx = p_importance.argsort()
feature_names = ['utterance_duration', 'pause_duration', 'duration_PVI', 'speech_rate', 'f0_mean', 'f0_range', 'f0_PVI']
#feature_names = ['utterance_duration', 'pause_duration', 'duration_PVI', 'speech_rate', 'f0_mean', 'f0_range', 'f0_PVI', 'f0_reset'] # considering f0-reset
features = np.array(feature_names)
plt.barh(features[sorted_idx], p_importance[sorted_idx])
plt.xlabel("Permutation Importance")


# In[ ]:


p_importance


# In[ ]:


#%%time

# text classifier （需跑將近50min)
#esem_t_cl = esemble_model.estimators_[0]
#from sklearn.inspection import permutation_importance
#r = permutation_importance(esem_t_cl, int2_x_test, int2_y_test,
#                           n_repeats=5,
#                           random_state=0, n_jobs=-1) ## `-1` run in parallel

# read permutation results
import pickle
with open('./esem_t_permutation.pkl', 'rb') as f:
    r = pickle.load(f)


for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:  ## importances_mean長度為1593(prosody 7 + text 1586)，前面7個features為prosodic features
        print(f"{vocab[i-7]:<8}" # 真正text featuresk的數值從r.importances_mean[:,7]開始，與原本vocab的index差7因此為vocab[i-7]
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")


# In[ ]:


# text importance plotting
t_importance = r.importances_mean[7:]  ## 只需要後面1586個text features的數值

# Select top 10
top_t_importance = np.argsort(t_importance)[::-1][:10][::-1]  ## [::-1]倒序輸出所有元素: 先由大到小排序，選前10個，再由小到大排序（畫圖y軸由小畫到大）

# plotting
import matplotlib
matplotlib.rcParams['font.family'] = ['Heiti TC'] # for chinese label
plt.barh(vocab[top_t_importance], t_importance[top_t_importance])
plt.xlabel("Permutation Importance")


# In[ ]:


# store permutation results
#import pickle
#with open('../PyLibrary/esem_t_permutation.pkl', 'wb') as f:
#    pickle.dump(r, f)


# In[ ]:


# final classifier
# weights assigned to the two sub-models
esem_final_cl = esemble_model.final_estimator_
print(esem_final_cl.n_features_in_)
print(esem_final_cl.coef_) # [text, prosody]


# # Chance level of informative speech

# In[ ]:


(2124+516)/(2124+1915+516+494)


# # Comparison: accuracy

# In[ ]:


print(p_model_forest.score(p_x_test, p_y_test)) ## best prosody-based
print(t_model_svm.score(x_test_bow, t_y_test))  ## best text-based
print(int_model_svm.score(int_x_test, t_y_test))  ## best integrated
print(esemble_model.score(int2_x_test, int2_y_test)) ## esemble


# # Comparison: precision

# In[ ]:


from sklearn.metrics import precision_score
y_pred_p_model_forest = p_model_forest.predict(p_x_test) ## best prosody-based
print(precision_score(p_y_test, y_pred_p_model_forest, 
         labels = None))

y_pred_t_model_svm = t_model_svm.predict(x_test_bow) ## best text-based
print(precision_score(t_y_test, y_pred_t_model_svm, 
         labels = None))

y_pred_int_model_svm = int_model_svm.predict(int_x_test) ## best integrated
print(precision_score(t_y_test, y_pred_int_model_svm, 
         labels = None))

y_pred_esemble_model = esemble_model.predict(int2_x_test) ## esemble
print(precision_score(int2_y_test, y_pred_esemble_model, 
         labels = None))


# # Comparison: recall

# In[ ]:


from sklearn.metrics import recall_score
y_pred_p_model_forest = p_model_forest.predict(p_x_test) ## best prosody-based
print(recall_score(p_y_test, y_pred_p_model_forest, 
         labels = None))

y_pred_t_model_svm = t_model_svm.predict(x_test_bow) ## best text-based
print(recall_score(t_y_test, y_pred_t_model_svm, 
         labels = None))

y_pred_int_model_svm = int_model_svm.predict(int_x_test) ## best integrated
print(recall_score(t_y_test, y_pred_int_model_svm, 
         labels = None))

y_pred_esemble_model = esemble_model.predict(int2_x_test) ## esemble
print(recall_score(int2_y_test, y_pred_esemble_model, 
         labels = None))


# # Comparison: f1 score

# In[ ]:


from sklearn.metrics import f1_score
y_pred_p_model_forest = p_model_forest.predict(p_x_test) ## best prosody-based
print(f1_score(p_y_test, y_pred_p_model_forest, 
         labels = None))

y_pred_t_model_svm = t_model_svm.predict(x_test_bow) ## best text-based
print(f1_score(t_y_test, y_pred_t_model_svm, 
         labels = None))

y_pred_int_model_svm = int_model_svm.predict(int_x_test) ## best integrated
print(f1_score(t_y_test, y_pred_int_model_svm, 
         labels = None))

y_pred_esemble_model = esemble_model.predict(int2_x_test) ## esemble
print(f1_score(int2_y_test, y_pred_esemble_model, 
         labels = None))


# # Error analysis

# In[ ]:


utts_test


# In[ ]:


t_y_test


# In[ ]:


# predictive vs true dataframe
err_df = pd.DataFrame(utts_test, 
             columns=['text'])

err_df['true_genre'] = t_y_test
err_df['svm_predictive_genre'] = t_model_svm.predict(x_test_bow)
err_df['ensem_predictive_genre'] = esemble_model.predict(int2_x_test)
err_df


# In[ ]:


# esemble correct but svm incorrect dataframe
txt_vs_ensem_df = err_df[err_df['true_genre'] == err_df['ensem_predictive_genre']]
txt_vs_ensem_df = txt_vs_ensem_df[txt_vs_ensem_df['ensem_predictive_genre'] != txt_vs_ensem_df['svm_predictive_genre']]
txt_vs_ensem_df


# In[ ]:


# store as csv
#txt_vs_ensem_df.to_csv('txt_vs_ensem_df.csv')

