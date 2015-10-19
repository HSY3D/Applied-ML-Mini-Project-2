# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 18:00:11 2015

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

'''
******************* GLOBAL VARIABLES *******************
'''
cachedStopWords = stopwords.words("english")
exclude = set(string.punctuation)
table = string.maketrans("","")

def main():
    x_train = readCSVX()
    y_train = readCSVY()
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(x_train)
    print count_vect.vocabulary_.get(u'and')
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_array = X_train_tfidf.toarray()
#    print getFreqDist(removeStopWordsAndPunct(x_train))
    zero, one, two, three = getTrainingExampleFrequency(x_train, y_train)
#    getFreqDist(removeStopWordsAndPunct(zero))
#    getFreqDist(removeStopWordsAndPunct(one))
#    getFreqDist(removeStopWordsAndPunct(two))
#    getFreqDist(removeStopWordsAndPunct(three))
#    print X_train_array
    print X_train_array[0][39225]
#    X_train_numpy = np.array(X_train_array)
#    X_train_numpy = X_train_numpy.T
#    print X_train_numpy[39225][0]
    print X_train_array.shape
    count=0
    for j in range(len(X_train_array)):
        if X_train_array[j][39225] != 0.0:       
#            print '%d: %f' % (j, X_train_array[j][39225])
            count += 1
    print count
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    total  = 0
    for val in y_train:
        total += 1
        if val == '0':
            count0 += 1
        elif val == '1':
            count1 += 1
        elif val == '2':
            count2 += 1
        elif val == '3':
            count3 += 1
    predictions = [count0, count1, count2, count3]
    parent_ent = entropy(predictions, total)
    print parent_ent
    ig_things = []
    ig_things.append(getInformationGain(39225, X_train_array, y_train, parent_ent))
          
'''
******************* HELPER FUNCTIONS *******************
'''

#give it a word('feature')
#get the nodes of the feature
#calculate their entropy individually
#calculate the information gain

def id3(examples, target_attr, attr):
    #Create roote node for the tree
    #if all the values in the data set are a single class, then return that class
    if len(examples[1]) == 0 and len(examples[2]) == 0 and len(examples[3]) == 0:
        return 0
    elif len(examples[0]) == 0 and len(examples[2]) == 0 and len(examples[3]) == 0:
        return 1
    elif len(examples[0]) == 0 and len(examples[1]) == 0 and len(examples[3]) == 0:
        return 2
    elif len(examples[0]) == 0 and len(examples[1]) == 0 and len(examples[2]) == 0:
        return 3
    else:
        #Choose the next best attribute to best classify our data
        best = choose_attr(examples, target_attr, attr)
        #Create a new decision tree/node with best attr and an empty dict object
        tree = {best:{}}
        #Create a new dec tree/sub-node for each of the values in the best attribute field
        for val in get_values(data,best):
            #Create a subtree for the current value under the "best" field
            subtree = id3(get_examples(data, best, val))
        

def getInformationGain(feature_num, x_data, y_vector, parent_entropy):
    attr_children = {}
    max_list = []
    nodes1 = []
    nodes2 = []
    nodes3 = []
    nodes0 = []
    for i in range(len(x_data)):
        tmp = x_data[i][feature_num]
        if tmp != 0.0:
            exp_str = str(i)
            attr_children[exp_str] = [tmp,y_vector[i]]
            max_list.append(tmp)
    max_value = max(max_list)
#    print attr_children
    thresholds = getThresholds(attr_children)
#    sort(thresholds)
    print thresholds
    for key, value in attr_children.iteritems():
        i = value[0]
        if abs(i-thresholds[0]) <= 0.005 and i <= thresholds[0]:
            nodes0.append(i)
        elif abs(i-thresholds[1]) <= 0.005 and i <= thresholds[1]:
            nodes1.append(i)
        elif abs(i-thresholds[2]) <= 0.005 and i <= thresholds[2]:
            nodes2.append(i)
        elif abs(i-thresholds[3]) <= 0.005 and i <= thresholds[3]:
            nodes3.append(i)
    total_count = len(attr_children)
    entropy_values = [entropy(nodes0, total_count), entropy(nodes1, total_count), entropy(nodes2, total_count), entropy(nodes3, total_count)] 
    nodes_counts = [len(nodes0),len(nodes1),len(nodes2),len(nodes3)]
    print nodes_counts    
    print entropy_values
    ig_sum = 0
    node_num = 0
    for e in entropy_values:
        ig_sum += (float(nodes_counts[node_num])/float(total_count))*float(e)
        node_num += 1
    print ig_sum
    return parent_entropy - ig_sum
        
#each frequency is the frequency of each word in each class   
def getThresholds(attr_children):
    thresholds = []
    nodes0 = []    
    nodes1 = []
    nodes2 = []
    nodes3 = []
    total = len(attr_children)
    for key, value in attr_children.iteritems():
        i = value[1]
        tmp = float(value[0])
        if i == '0':
            nodes0.append(tmp)
        elif i == '1':
            nodes1.append(tmp)
        elif i == '2':
            nodes2.append(tmp)
        elif i == '3':
            nodes3.append(tmp)
    print len(nodes0)
    print len(nodes1)
    print len(nodes2)
    print len(nodes3)
    sum=float(0)
    for i in nodes0:
        sum += float(i)
    print sum
    sum = float(sum)/float(len(nodes0))
    thresholds.append(sum)
#    thresholds['0'] = sum
    sum=float(0)
    for i in nodes1:
        sum += float(i)
    print sum
    sum = float(sum)/float(len(nodes1))
    thresholds.append(sum)
#    thresholds['1'] = sum
    sum=float(0)
    for i in nodes2:
        sum += float(i)
    print sum
    sum = float(sum)/float(len(nodes2))
    thresholds.append(sum)
#    thresholds['2'] = sum
    sum=float(0)
    for i in nodes3:
        sum += float(i)
    print sum
    sum = float(sum)/float(len(nodes3))
    thresholds.append(sum)
#    thresholds['3'] = sum
    return thresholds
    
def entropy(predictions, total):
    entropyVal = 0
    for count in predictions:
        tmp = float(count)/float(total)
        entropyVal += (tmp)*math.log(tmp,2)
    return -entropyVal

def getFreqDist(data):
    data = ' '.join(data)
    fdist = FreqDist(word.lower() for word in word_tokenize(data))
    print fdist
    
def calculateIG():
    return 0
    
def getTrainingExampleFrequency(x, y):
    i = 0 
    zero = []
    one = []
    two = []
    three = []
    for j in y:
        sent = x[i]
        if j == '0':
            zero.append(sent)
        elif j == '1':
            one.append(sent)
        elif j == '2':
            two.append(sent)
        elif j == '3':
            three.append(sent)
        i += 1

    return zero, one, two , three
            

def savetocsv(urls):
    with open('values.csv', "w") as output:
        writer = csv.writer(output, lineterminator=',')
        for url in urls:
            writer.writerow([url])
    print 'saved'

def removeStopWordsAndPunct(data):
    refined_data = []    
    for exmpl in data:
        exmpl = exmpl.translate(table, string.punctuation)
        exmpl = ' '.join([word for word in exmpl.split() if word.lower() not in cachedStopWords])
        refined_data.append(exmpl)
    return refined_data

def readCSVX():
    data = pd.read_csv('ml_trainingset.csv',quotechar = '"').as_matrix()
    x_test = data[:,1]
    x_test= x_test.astype(str)
    return x_test
    
def readCSVY():
    data = pd.read_csv('ml_trainingset.csv',quotechar = '"').as_matrix()
    x_test = data[:,2]
    x_test= x_test.astype(str)
    return x_test
 
def tokenizeSentences(text):
    return [word_tokenize(text)]

main()