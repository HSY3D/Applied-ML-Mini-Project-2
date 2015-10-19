# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:43:17 2015

@author: hannansyed
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import pandas as pd
import string
import csv
import numpy as np
import math


def main():
    #Read data
    x_train = readCSV('x')
    y_train = readCSV('y')
    
    #Get IDF of data
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(x_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_array = X_train_tfidf.toarray()
    
    RocchioTraining(X_train_array, y_train)
#    id3Tree(X_train_array, y_train, attr_num)
    
def id3Tree(data, targets, attr_num):
    #if all the values in the data set are a single class, then return that class
    #Choose the next best attribute to best classify our data
    best = getBest(data, targets, attr_num, parent_entropy)
    #Create a new decision tree/node with best attr and an empty dict object
    #Create a new dec tree/sub-node for each of the values in the best attribute field
    #Create a subtree for the current value under the "best" field
    return 0

def getBest(data, targets, attr_num):
    for word in data:
        getEntropy(word, data, targets, parent_entropy)
    

def getEntropy(feature_num, x_data, y_vector, parent_entropy):
    attr_children = {}
    nodes0=[]
    nodes1=[]
    nodes2=[]
    nodes3=[]
    
    for document_num in range(len(x_data)):
        tmp = data[document_num][feature_num]
        if tmp != 0.0:
            attr_children[str(document_num)] = [tmp, y_vector[i]]
    
    thresholds = getThresholds(attr_children)
    
#conditional frequency
def getThresholds(attr_children):
    return 0
    
def RocchioTraining(data, y_data):
    prototype0,prototype1,prototype2,prototype3=np.zeros(len(data[0])),np.zeros(len(data[0])),np.zeros(len(data[0])),np.zeros(len(data[0]))
    for index in range(len(data)):
        d = np.array(data[index])
        c = y_data[index]
        if c == '0':
            prototype0 = np.add(prototype0,d)
        elif c == '1':
            prototype1 = np.add(prototype1,d)
        elif c == '2':
            prototype2 = np.add(prototype2,d)
        elif c == '3':
            prototype3 = np.add(prototype3,d)
    
def readCSV(strxy):
    if strxy == 'x':
        data = pd.read_csv('ml_trainingset.csv',quotechar = '"').as_matrix()
        x_test = data[:,1]
        x_test= x_test.astype(str)
        return x_test
    else:
        data = pd.read_csv('ml_trainingset.csv',quotechar = '"').as_matrix()
        y_test = data[:,2]
        y_test= y_test.astype(str)
        return y_test
    
main()
    
    
