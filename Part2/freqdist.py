# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:12:46 2015

@author: hannansyed
"""

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import pandas as pd

def main():
    data = readCSV()
    for i in data:
        fdist = FreqDist(word.lower() for word in word_tokenize(i))
    print fdist
    print fdist.most_common(50)

def readCSV():
    data = pd.read_csv('ml_trainingset.csv',quotechar = '"').as_matrix()
    x_test = data[:,1]
    x_test= x_test.astype(str)
    return x_test
    
main()
