
import numpy as np
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn import cross_validation  
from scipy.sparse import csr_matrix 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Extract file living at "path". Only IDs present in included set will be extracted.
# If we are using the testing data we are short 1 column so we take a different code path.
# returns an n x 3 matrix; (ID, Interview, Classification) and the number of articles in the path.

#### Has to be a list when you do a grid search 


svm_flag = 1
sgd_flag = 0
bin_flag = 0
num_example = 20000

param_svm = {'vect__ngram_range': [(1, 1),(1,2)],
    }

## To change because this has to be a list 
param_sgd = {'vect__ngram_range': [(1, 1), (1, 2),(1,3),(1,4)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-1, 1e-3),
               'clf__loss': ('hinge','log'),
               'clf__penalty': ('l2','elasticnet'),
    }
param_bin = {'vect__ngram_range': [(1, 1), (1, 2),(1,3),(1,4)],
               'tfidf__use_idf': (True, False),
    }



if __name__ == '__main__':

    data = pd.read_csv('./trainingdata/ml_trainingset.csv',quotechar = '"').as_matrix()
    split = data.shape[0]*0.8
    x = data[:split,1]
    y = data[:split,2]
    x_val = data[split:,1]
    y_val = data[split:,2]
    x = x.astype(str)
    y = y.astype(int)
    x_val = x_val.astype(str)
    y_val = y_val.astype(int)

    ############## need to do a serious parameter search for kbest
    if svm_flag == 1:
        text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                             ('tfidf', TfidfTransformer(use_idf = True)),
                             ('clf', LinearSVC()),])
        parameters = param_svm

    if sgd_flag == 1:
        text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier(random_state = 42,n_iter = 1)),])
        parameters = param_sgd

    if bin_flag == 1:
        text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),])
        parameters = param_bin



    gs_clf = GridSearchCV(text_clf,parameters,n_jobs = 7)
    gs_clf = gs_clf.fit(x,y)
    #scores = cross_validation.cross_val_score(text_clf,x,y,cv = 2)
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    print score
    pred = gs_clf.predict(x_val)
    print np.mean(pred == y_val)
    print (metrics.classification_report(y_val, pred,target_names=['0','1','2','3']))
    print metrics.confusion_matrix(y_val, pred)
    '''
    tolkenizedSentences = parse_sentences(extractedData, fileLength)
    extractedData = np.hstack((tolkenizedSentences.reshape(fileLength, 1), extractedData[:, 2].reshape(fileLength, 1)))
    del tolkenizedSentences
    # Create features and our predictor
    featureHash, featureSpace, featureNum = create_ngrams(extractedData[:, 0])
    classes = extractedData[:, 1]
    featureSpace = csr_matrix(featureSpace)
    del extractedData
    #predictor = create_naive_bayes_predictor(featureSpace, featureNum, classes)
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_validation.cross_val_score(clf,featureSpace,classes,cv = 2)
    print scores
    del featureSpace

scp -r text_class_search.py  pthodo@agent-server.cs.mcgill.ca:/home/ml/pthodo/comp_598/assignement_2/COMP598A2/text_class_search.py    '''