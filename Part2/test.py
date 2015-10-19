# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:53:59 2015

@author: hannansyed
"""
import pandas as pd
import math 

def main():
    data = pd.read_csv('ml_trainingset.csv',quotechar = '"').as_matrix()
    y_test = data[:,2]
    y_test= y_test.astype(str)
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    total  = 0
    for val in y_test:
        total += 1
        if val == '0':
            count0 += 1
        elif val == '1':
            count1 += 1
        elif val == '2':
            count2 += 1
        elif val == '3':
            count3 += 1
    print count0
    print count1
    print count2
    print count3
    print total
    predictions = [count0, count1, count2, count3]
    print entropy(predictions, total)
    
def entropy(predictions, total):
    entropyVal = 0
    for count in predictions:
        tmp = float(count)/float(total)
        entropyVal += (tmp)*math.log(tmp,2)
    return -entropyVal

def informationGain(feature, prediction):
    #null
    return 0

def getThresholds(max_val):
    percentages = [0.2, 0.4, 0.6, 0.8]
    thresholds = []
    for p in percentages:
        thresholds.append(max_val*p)
    return thresholds
#    nodes1 = []
#    nodes2 = []
#    nodes3 = []
#    nodes0 = []
#    for key, value in attr_children.iteritems():
#        i = value[1]
#        if i == 0:
#            nodes0.append(i)
#        elif i == 1:
#            nodes1.append(i)
#        elif i == 2:
#            nodes2.append(i)
#        elif i == 3:
#            nodes3.append(i)
    
    
main()