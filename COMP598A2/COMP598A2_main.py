__author__ = 'erickissel'

import numpy as np
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Extract file living at "path". Only IDs present in included set will be extracted.
# If we are using the testing data we are short 1 column so we take a different code path.
# returns an n x 3 matrix; (ID, Interview, Classification) and the number of articles in the path.

def extract_file(path, includedSet = [], usingTestData = False):
    regex = r"(\d+),\"(.+?)\",(\d+)"
    if usingTestData:
        regex = r"(\d+),\"(.+?)\""
    file = open(path)
    ids = []
    interviews = []
    result = []
    i = 0
    for line in file:
        if usingTestData:
            m = re.search(regex, line)
            if not m is None:
                ids.append(m.group(1))
                interviews.append((m.group(2)))
            else:
                ids.append([i])
                interviews.append([])

        else:
            m = re.search(regex, line)
            if not m is None and i in includedSet:
                ids.append(m.group(1))
                interviews.append((m.group(2)))
                result.append((m.group(3)))
        i += 1

    fileLength = len(ids)
    ids = np.array(ids).reshape([fileLength, 1])
    interviews = np.array(interviews).reshape([fileLength, 1])
    if not usingTestData:
        result = np.array(result).reshape([fileLength, 1])
    if not usingTestData:
        data = np.hstack((ids, interviews, result))
        return data, fileLength
    else:
        data = np.hstack((ids,interviews))
        return data, fileLength
    return data, fileLength

# Produces unigram features and then populates an array of size (NumberOfArticles x NumberOfFeatures) with the appropriate data.
# Returns this array of features.

def create_ngrams(array, classifier):
    featureHash = {}
    featureIndex = 0
    for sent in array:
        for word in sent:
            if not featureHash.has_key(word):
                featureHash[word] = featureIndex
                featureIndex += 1
    featureSpace = np.zeros([4, featureIndex], 'f')
    for i in range(0, len(array)):
        for word in array[i]:
            featureSpace[int(classifier[i])][featureHash[word]] += 1

    return featureHash, featureSpace, featureIndex

def features_of_interview(sentence, featureLength, featureHash):
    sentence_features = np.zeros(featureLength)
    for word in sentence:
        if featureHash.has_key(word):
            sentence_features[featureHash[word]] += 1
    return sentence_features

# Generates the bayes predictor given the featureSpace of our training set, the number of features and the corresponding classes
# these interviews fall in to.

def create_naive_bayes_predictor(featureSpace, featureLength, classes):
    predictor = np.zeros([1, featureLength + 1])
    for i in range (0, 4):
        tempArray = featureSpace[i] + 0.1
        tempArray.reshape(1, featureLength)
        val = np.sum(tempArray, axis=0)
        height = classes[classes[:] == '%s' % i].size
        tempArray.reshape(1, featureLength)
        tempArray = tempArray / val
        tempArray = np.hstack((tempArray, np.array([(height + 0.0)/classes.size])))
        if i == 0:
            predictor = tempArray
        else:
            predictor = np.vstack((predictor, tempArray))
    return predictor

# Parse each sentence. Note that word manipulation (i.e. stemming) should be done here.

def parse_sentences(extractedData, fileLength):
    tolkenizedWords = []
    for i in range (0, fileLength):
        try:
            nextAddition = word_tokenize(extractedData[i, 1])
            tolkenizedWords.append(nextAddition)
        except:
            print ('Failed to tolkenize sentence %d' % i)
            tolkenizedWords.append([])
    return np.array(tolkenizedWords)

# Given our predictor, the number of features we generated, the hash h : word -> int and a sentence to predict
# We output a predictor.

def predict_interview_class(class_predictor, sentence, featureLength, featureHash):
    logPredictor = np.log(class_predictor)                                                  # Move this outside.
    featuresOfSentence = features_of_interview(sentence, featureLength, featureHash)
    ones = np.array([1])
    ones.reshape(1,1)
    featuresOfSentence = np.hstack((featuresOfSentence, ones))
    featuresOfSentence.reshape(1, featureLength + 1)
    logPrediction = logPredictor * featuresOfSentence
    weights = np.sum(logPrediction, axis=1)

    max = weights[0]
    index = 0
    for j in range(1, 4):
        if weights[j] > max:
            max = weights[j]
            index = j
    return index


if __name__ == '__main__':

    # Get training data

    extractedData, fileLength = extract_file('./trainingdata/ml_trainingset.csv', range(0, 60000))
    tolkenizedSentences = parse_sentences(extractedData, fileLength)
    extractedData = np.hstack((tolkenizedSentences.reshape(fileLength, 1), extractedData[:, 2].reshape(fileLength, 1)))
    del tolkenizedSentences
    # Create features and our predictor

    featureHash, featureSpace, featureNum = create_ngrams(extractedData[:, 0], extractedData[:,1])
    classes = extractedData[:, 1]
    del extractedData
    predictor = create_naive_bayes_predictor(featureSpace, featureNum, classes)
    print predictor
    del featureSpace
    
    # Extract test data
    testData, testLength = extract_file('./trainingdata/ml_testingset.csv', [], True)
    tokenizedData = parse_sentences(testData, testLength)

    # Open prediction file
    predictionsFile = open('./trainingdata/predictions.csv', 'w')
    predictionsFile.write('Id,Prediction\n')
    currentIndex = -1

    # Make predictions
    for sentence in tokenizedData:
        if currentIndex == -1:
            currentIndex += 1
            continue
        nextPrediction = predict_interview_class(predictor, sentence, featureNum, featureHash)
        predictionsFile.write('%d,%d\n' % (currentIndex,nextPrediction))
        currentIndex += 1





