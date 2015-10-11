
import numpy as np
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
from sklearn import svm
import numpy as np
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
# Extract file living at "path". Only IDs present in included set will be extracted.
# If we are using the testing data we are short 1 column so we take a different code path.
# returns an n x 3 matrix; (ID, Interview, Classification) and the number of articles in the path.

#### Has to be a list when you do a grid search 


svm_flag = 1
sgd_flag = 0
bin_flag = 0
ada_flag = 0
num_example = 60000
num_test_example = 6000
param_svm = {'vect__ngram_range': [(1, 1),(1,2),(1,3)],
               'tfidf__use_idf': (True,False),
               'clf__kernel': ('linear', 'rbf'),
    }

## To change because this has to be a list 
param_sgd = {'vect__ngram_range': [(1, 1), (1, 2),(1,3),(1,4)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-1, 1e-3),
               'clf__loss': ('hinge','log'),
               'clf__penalty': ('l2','elasticnet'),
               'clf__n_iter': (1,5),
               'clf__random_state' : (42,52),
    }
param_bin = {'vect__ngram_range': [(1, 1), (1, 2),(1,3),(1,4)],
               'tfidf__use_idf': (True, False),
    }

def print_submission(prediction):
    id_list = np.arange(prediction.shape[0])
    print prediction
    #,comments= '',header = 'Id,Prediction'
    np.savetxt('submission.csv',np.c_[id_list.astype(int), prediction.astype(int)],delimiter=',')

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

   # x_test = extractedData[:split,1]
   #y_test = extractedData[:split,2]
    if ada_flag == 1:
        text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                             ('tfidf', TfidfTransformer()), 
                             ('clf', RandomForestClassifier(n_estimators = 20)),])
    if svm_flag == 1:
        #('feat', SelectKBest(chi2,k = 20000)), put it back later

        text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english',ngram_range = (1,2))),
                             ('tfidf', TfidfTransformer()), 
                             ('clf', LinearSVC(penalty = 'l2',dual = False, tol = 1e-4)),])
    if sgd_flag == 1:
        text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier(loss='hinge',
                                                    penalty='l2',                                            
                                                    alpha=1e-3,
                                                    n_iter=5,
                                                    random_state=42)),])
    if bin_flag == 1:
        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),])


    text_clf = text_clf.fit(x,y)
    pred = text_clf.predict(x_val)

    print np.mean(pred == y_val)
    print (metrics.classification_report(y_val, pred,target_names=['0','1','2','3']))
    print metrics.confusion_matrix(y_val, pred)
    data = pd.read_csv('./trainingdata/ml_testingset.csv',quotechar = '"').as_matrix()

    x_test = data[:,1]
    x_test= x_test.astype(str)


    pred = text_clf.predict(x_test).astype(int)
    print_submission(pred)


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
    '''