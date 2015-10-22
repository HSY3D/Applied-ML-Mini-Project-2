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

X_train_array = 0
def main():
    global X_train_array
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
#    print get_values(39225, X_train_array, y_train)
    attr=[29041,8955,48913,41334,31837,37629,37458,38097,19981,46715,47110,29985,8643,45790,13892,27640,27726,32415,40287,22617]
    attributes = []
    for i in range(len(X_train_array[0])):
        attributes.append(i)
#    best =  getBest(X_train_array, y_train, 39225)
#    print best
#    print attributes
#    initial_ex = [example for example in X_train_array]
    initial_ex={}
    for i in range(len(X_train_array)):
        tmp = X_train_array[i][2512]
        if tmp != 0.0:
            initial_ex[i] = tmp
#    initialbest = getBest(initial_ex, attr, x_train, y_train, 36012)
#    print initialbest
#    print majority_value(X_train_array, y_train, 39225)
#    tree = id3Tree(initial_ex, X_train_array, y_train, attributes, [2512,0.032])
#    print tree
    print get_values(initial_ex.keys(),attr, 36012, X_train_array, y_train)
#    print idTree(initial_ex, [2512,0.032], attributes, X_train_array, y_train, 0)
    
    
def id3Tree(examples, x_train, y_train, attributes, target_attr):
    global X_train_array
    x_train = x_train[:]
    y_train = y_train[:]
    default = majority_value(examples.keys(), attributes, x_train, y_train, target_attr[0])
    nodes = get_values(examples.keys(), attributes, target_attr[0], x_train, y_train)
    print 'here'
    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not examples or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif len(nodes[0]) == 1 and len(nodes[1]) == 0 and len(nodes[2]) == 0 and len(nodes[3]) == 0:
        i = nodes[0].keys()
        d = nodes[0]      
        return {target_attr[0]:d[i[0]][1]}
    elif len(nodes[1]) == 1 and len(nodes[0]) == 0 and len(nodes[2]) == 0 and len(nodes[3]) == 0: 
        i = nodes[0].keys()
        d = nodes[0]        
        return {target_attr[0]:d[i[0]][1]}
    elif len(nodes[2]) == 1 and len(nodes[0]) == 0 and len(nodes[1]) == 0 and len(nodes[3]) == 0: 
        i = nodes[0].keys()
        d = nodes[0]        
        return {target_attr[0]:d[i[0]][1]}
    elif len(nodes[3]) == 1 and len(nodes[0]) == 0 and len(nodes[2]) == 0 and len(nodes[1]) == 0: 
        i = nodes[0].keys()
        d = nodes[0]        
        return {target_attr[0]:d[i[0]][1]}
    else:
        #Choose the next best attribute to best classify our data
        best = getBest(examples, attributes, x_train, y_train, target_attr[0])
        feature_num = best[0]
        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {best[0]:{}}
        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        node_num=0
        values = get_values(examples, attributes, int(feature_num), x_train, y_train)
        for node in values:
            # Create a subtree for the current value under the "best" field
            subtree = id3Tree(node, x_train, y_train, [attr for attr in attributes if attr != best], best)
            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
#            print subtree
            tree[best[0]][node_num] = subtree
            node_num += 1
    return tree

def idTree(examples, target_attr, attributes, x_train, y_train, k):
#    Create a root node for the tree
#    If all examples are positive, Return the single-node tree Root, with label = +.
#    If all examples are negative, Return the single-node tree Root, with label = -.
#    If number of predicting attributes is empty, then Return the single node tree Root,
#    with label = most common value of the target attribute in the examples.
#    Otherwise Begin
#        A ← The Attribute that best classifies examples.
#        Decision Tree attribute for Root = A.
#        For each possible value, vi, of A,
#            Add a new tree branch below Root, corresponding to the test A = vi.
#            Let Examples(vi) be the subset of examples that have the value vi for A
#            If Examples(vi) is empty
#                Then below this new branch add a leaf node with label = most common target value in the examples
#            Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
#    End
#    Return Root
    tree = {target_attr[0]:{}}
    print tree
    if k == 3:
        return tree
    best = getBest(examples, attributes, x_train, y_train, int(target_attr[0]))
    node_num=0
    for node in get_values(examples, attributes, target_attr[0], x_train, y_train):
        subtree = idTree(node, best, [attr for attr in attributes if attr != best[0]], x_train, y_train,(k+1))
        tree[best[0]][node_num] = subtree
        node_num+=1
    return tree
    
def get_values(examples, attributes, feature_num, x_data, y_vector):
    nodes0={}
    nodes1={}
    nodes2={}
    nodes3={}
    
    vector=[]
    for i in examples:
        tmp = x_data[i][int(feature_num)]
        if tmp != 0.0 or (tmp not in attributes):
            vector.append(tmp)        
    t1,t2,t3 = getT3(vector)
    
#    print t2
    for document_num in examples:
        tmp = x_data[document_num][int(feature_num)]
        if tmp != 0.0 or (tmp not in attributes):
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

def majority_value(examples, attributes,x_train, y_train, target_attr):
    x_train = x_train[:]
    y_train = y_train[:]
    
    count = [0,0,0,0]
    for i in range(len(x_train)):
        if x_train[i][int(target_attr)] != 0.0 or x_train[i][int(target_attr)] not in attributes:
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

def getBest(example, attributes, x_train, y_train, target_attr):
    igs={}
    for i in range(len(x_train)):
        tmp = x_train[i][int(target_attr)]
        if tmp != 0.0 or (tmp not in attributes):
            ig = getInformationGain(i, x_train, y_train)
            igs[str(i)] = float(ig)
#        print ig
    
    return max(igs.iteritems(), key=operator.itemgetter(1))

def getInformationGain(feature_num, x_data, y_vector):
    nodes0={}
    nodes1={}
    nodes2={}
    nodes3={}
    
    vector=[]
    for i in range(len(x_data)):
        tmp = x_data[i][int(feature_num)]       
        if tmp != 0.0:
            vector.append(tmp)
            
    t1,t2,t3 = getT2(vector)
    
    for document_num in range(len(x_data)):
        tmp = x_data[document_num][int(feature_num)]
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
    
def getT3(vector):
    stddiv=0
    vector = np.array(vector)
    stddiv = np.average(vector)
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
    
    
