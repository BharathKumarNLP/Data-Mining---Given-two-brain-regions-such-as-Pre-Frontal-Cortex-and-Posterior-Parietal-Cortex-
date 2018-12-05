# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 00:06:44 2018

@author: Bharath Kumar N
"""


import pandas as pd 
import sklearn
import numpy as np
import nltk
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn import tree
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import chi2

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support


import gensim, logging
from gensim.models import Word2Vec
from scipy import sparse

def loadData(filePath="dataset.csv"):
    data = pd.read_csv(filePath, header=0)
    return data["Pre Frontal Cortex"],data["Posterior Parietal Cortex"]

def preProcessing(features):
    num_titles = features.size
    clean_wordlist = []
    clean_titles = []
    stops = set(stopwords.words('english'))
    for i in range( 0, num_titles):
        #letters_only = re.sub("[^a-zA-Z]", " ", features[i]) 
        words = features[i].lower().split()
        words = [w.lower() for w in words if not w in stops]  
        clean_wordlist.append(words)
        clean_titles.append(" ".join(words))
    return clean_titles, clean_wordlist

def getDTMByTFIDF(features,nfeatures):
    tfIdf_vectorizer = TfidfVectorizer(max_features=nfeatures)
    dtm = tfIdf_vectorizer.fit_transform(features).toarray()
    return dtm,tfIdf_vectorizer

def featuresByChiSq(features,labels,nFeature=5000):
    chi2_model = SelectKBest(chi2,k=nFeature)
    dtm = chi2_model.fit_transform(features,labels)
    return dtm,chi2_model

def featuresByInformationGain(features,labels):
    treeCL = tree.DecisionTreeClassifier(criterion="entropy")
    treeCL = treeCL.fit(features,labels)
    transformed_features = SelectFromModel(treeCL,prefit=True).transform(features)
    return transformed_features

def featuresByLSA(features,ncomponents=100):
    svd = TruncatedSVD(n_components=ncomponents)
    normalizer =  Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    dtm_lsa = lsa.fit_transform(features)
    return dtm_lsa

def makeFeatureVec(words, model, num_features):
    feature_vec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec,model[word]) 

    feature_vec = np.divide(feature_vec,nwords)
   
    return feature_vec

def getAvgFeatureVecs(title, model, num_features):
    counter = 0.
    titleFeatureVecs = np.zeros((len(title), num_features),dtype="float32")
    for t in title:
        titleFeatureVecs[counter] = makeFeatureVec(t, model,num_features)
        counter = counter + 1.
    return titleFeatureVecs


def crossValidate(document_term_matrix,labels,classifier="SVM",nfold=2):
    clf = None
    precision = []
    recall = []
    fscore = []
    
    if classifier == "RF":
        clf = RandomForestClassifier()
    elif classifier == "NB":
        clf = MultinomialNB()
    elif classifier == "SVM":
        clf = LinearSVC()
    
    skf = StratifiedKFold(labels, n_folds=nfold)

    for train_index, test_index in skf:
        X_train, X_test = document_term_matrix[train_index], document_term_matrix[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        model = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        p,r,f,s = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        
    return np.mean(precision),np.mean(recall),np.mean(fscore)

titles, labels = loadData()
processed_titles, processed_titles_wordlist = preProcessing(titles)
dtm,vect = getDTMByTFIDF(processed_titles,None)

chisqDtm, chisqModel = featuresByChiSq(dtm,labels,2000)
#igDtm = featuresByInformationGain(dtm,labels)
#lsaDtm = featuresByLSA(dtm,100)

num_features = 300    # Word vector dimensionality                      
min_word_count = 1    # Minimum word count                        
num_workers = 1       # Number of threads to run in parallel
context = 8           # Context window size    

downsampling = 1e-5   # Downsample setting for frequent words

word2vec_model = Word2Vec(processed_titles_wordlist, workers=num_workers, 
            size=num_features, min_count = min_word_count, 
            window = context, sample = downsampling)
word2vec_model.init_sims(replace=True)


wordVecs = getAvgFeatureVecs(processed_titles_wordlist, word2vec_model, num_features)

#Combine features from chiSq and word2Vec
combinedFeatures = np.hstack([chisqDtm,wordVecs])

precision, recall, fscore = crossValidate(chisqDtm,labels,"SVM",10)
print "ChiSq Features:",precision, recall, fscore

precision, recall, fscore = crossValidate(combinedFeatures,labels,"SVM",10)
print "ChiSq Features:",precision, recall, fscore

