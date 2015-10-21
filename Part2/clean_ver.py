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
import operator


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
    
#    getThresholds(X_train_array, y_train,39225)
#    RocchioTraining(X_train_array, y_train)
#    id3Tree(X_train_array, y_train, attr_num)
#    getT2(X_train_array,39225)
    print getBest(X_train_array, y_train, 39225, 0)
    
def id3Tree(examples, y_train,attributes, target_attr, parent_entropy):
    examples = examples[:]
#    vals = [record[target_attr] for record in data]
    default = majority_value(data, target_attr)
    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not examples or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif len(examples[1]) == 0 and len(examples[2]) == 0 and len(examples[3]) == 0:
        return 0
    elif len(examples[0]) == 0 and len(examples[2]) == 0 and len(examples[3]) == 0:
        return 1
    elif len(examples[0]) == 0 and len(examples[1]) == 0 and len(examples[3]) == 0:
        return 2
    elif len(examples[0]) == 0 and len(examples[1]) == 0 and len(examples[2]) == 0:
        return 3
    else:
        #Choose the next best attribute to best classify our data
        best = getBest(examples, attributes, target_attr, parent_entropy)
        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {best:{}}
        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for nodes in get_values(x_train, y_train, best, parent_entropy):
            # Create a subtree for the current value under the "best" field
            subtree = id3Tree(nodes, [attr for attr in attributes if attr != best], target_attr)
            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree
    return tree

def majority_value(x_train, y_train, target_attr):
    x_train = x_train[:]
    y_train = y_train[:]
    count = [0,0,0,0]
    for i in range(len(x_train)):
        if x_train[i] != 0.0:
            if y_train[i] == '0':
                count[0] += 1
            elif y_train[i] == '1':
                count[1] += 1
            elif y_train[i] == '2':
                count[2] += 1
            elif y_train[i] == '3':
                count[3] += 1
    for n in count:
        if count[0] > count[1] and count[0] > count[2] and count[0] > count[3]:
            return 0
        if count[1] > count[0] and count[1] > count[2] and count[1] > count[3]:
            return 1
        if count[2] > count[0] and count[2] > count[1] and count[2] > count[3]:
            return 2
        if count[3] > count[0] and count[3] > count[1] and count[3] > count[2]:
            return 3

def getBest(x_train, y_train, target_attr, parent_entropy):
    igs={}
    for i in range(2000):
        ig = getInformationGain(i, x_train, y_train, parent_entropy)
        igs[str(i)] = float(ig)
        print ig
    
    return max(igs.iteritems(), key=operator.itemgetter(1))[0]

def get_values(feature_num, x_data, y_vector, parent_entropy):
    nodes0={}
    nodes1={}
    nodes2={}
    nodes3={}
    
    vector=[]
    for i in range(len(x_data)):
        tmp = x_data[i][feature_num]       
        if tmp != 0.0:
            vector.append(tmp)
            
    t1,t2,t3 = getT2(vector)
    
    for document_num in range(len(x_data)):
        tmp = x_data[document_num][feature_num]
        if tmp != 0.0:
#            attr_children[str(document_num)] = [tmp, y_vector[document_num]]
            if tmp <= t1:
                nodes0[str(document_num)] = [tmp,y_vector[document_num]]
            elif t1 < tmp and tmp <= t2:
                nodes1[str(document_num)] = [tmp,y_vector[document_num]]
            elif t2 < tmp and tmp < t3:
                nodes2[str(document_num)] = [tmp,y_vector[document_num]]
            elif tmp >= t3:
                nodes3[str(document_num)] = [tmp,y_vector[document_num]]
                
    return [nodes0,nodes1,nodes2,nodes3]

def getInformationGain(feature_num, x_data, y_vector, parent_entropy):
    nodes0={}
    nodes1={}
    nodes2={}
    nodes3={}
    
    vector=[]
    for i in range(len(x_data)):
        tmp = x_data[i][feature_num]       
        if tmp != 0.0:
            vector.append(tmp)
            
    t1,t2,t3 = getT2(vector)
    
    for document_num in range(len(x_data)):
        tmp = x_data[document_num][feature_num]
        if tmp != 0.0:
#            attr_children[str(document_num)] = [tmp, y_vector[document_num]]
            if tmp <= t1:
                nodes0[str(document_num)] = [tmp,y_vector[document_num]]
            elif t1 < tmp and tmp <= t2:
                nodes1[str(document_num)] = [tmp,y_vector[document_num]]
            elif t2 < tmp and tmp < t3:
                nodes2[str(document_num)] = [tmp,y_vector[document_num]]
            elif tmp >= t3:
                nodes3[str(document_num)] = [tmp,y_vector[document_num]]
#    print "%d,%d,%d,%d" %(len(nodes0),len(nodes1),len(nodes2),len(nodes3))
    total_count = len(vector)
    entropy_values = [entropy(nodes0, total_count), entropy(nodes1, total_count), entropy(nodes2, total_count), entropy(nodes3, total_count)]
    nodes_counts = [len(nodes0),len(nodes1),len(nodes2),len(nodes3)]
#    print entropy_values
    ig_sum = 0
    node_num = 0
    for e in entropy_values:
        ig_sum += (float(nodes_counts[node_num])/float(total_count))*float(e)
        node_num += 1
#    print "ig_sum:%f" %(ig_sum)
    return ig_sum


def entropy(node, total):
    entropyVal,count0,count1,count2,count3 = 0,0,0,0,0
    for key, val in node.iteritems():
        if val[1] == '0':
            count0 += 1
        elif val[1] == '1':
            count1 += 1
        elif val[1] == '2':
            count2 += 1
        elif val[1] == '3':
            count3 += 1
    predictions = [count0, count1, count2, count3]
#    print predictions
    for count in predictions:
        if count == 0:
            continue
        tmp = float(count)/float(total)
#        print tmp
        entropyVal += (tmp)*math.log(tmp,2)
    return -entropyVal
    
def getT2(vector):
    stddiv=0
    vector = np.array(vector)
    stddiv = np.std(vector)
#    print stddiv
    t0=(0.5)*stddiv
    t1=2*stddiv
    return t0,stddiv,t1
    
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
    
    
