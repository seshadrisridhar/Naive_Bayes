#!/usr/bin/python
#encoding:utf-8

import csv
import random
import math

#Handle Data
def LoadCsv(filename):
    lines = csv.reader(open(filename,"rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

#split a given dataset into a given split ratio.
#trainingdata and testdata
def splitDataset(dataset,splitRatio):
    trainSize = int(len(dataset)*splitRatio)
    trainSet  = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

#get mean and standard deviation for each attribute
def summarize(dataset):
    summarizes = [(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summarizes[-1]
    return summarizes

#calculate statistics for each class
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue,instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries

#calculate Gaussian probability for x
def calculateProbability(x,mean,stdev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

#calculate Class Probabilites
def calculateClassProbabilities(summaries,inputVector):
    probabilies = {}
    for classValue,classSummaries in summaries.iteritems():
        probabilies[classValue] = 1
        for i in range(len(classSummaries)):
            mean,stdev = classSummaries[i]
            x = inputVector[i]
            probabilies[classValue] *= calculateProbability(x,mean,stdev)
    return probabilies

#make a prediction
def predict(summaries,inputVector):
    probabilites = calculateClassProbabilities(summaries,inputVector)
    bestLabel,bestProb = None,-1
    for classValue,probability in probabilites.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries,testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries,testSet[i])
        predictions.append(result)
    return predictions

#calculate accuracy ratio
def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0


def main():
    #filename = "pima-indians-diabetes.data.csv"
    #dataset  = LoadCsv(filename)
    #print("Loaded data file {0} with {1} rows").format(filename,len(dataset))
    
    #dataset = [[1],[2],[3],[4],[5]]
    #splitRatio = 0.67
    #train,test = splitDataset(dataset,splitRatio)
    #print("Split {0} rows into train with {1} and test with {2}").format(len(dataset),train,test)
    
    #dataset = [[1,20,1],[2,21,0],[3,22,1]]
    #separated = separateByClass(dataset)
    #print('Separated instances:{0}').format(separated)
    
    #numbers = [1,2,3,4,5]
    #print('Summary of {0}: mean = {1},stdev = {2}').format(numbers,mean(numbers),stdev(numbers))
    
    #dataset = [[1,20,0],[2,21,1],[3,22,0]]
    #summary = summarize(dataset)
    #print("Attribute summaries: {0}").format(summary)
    
    #dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
    #summary = summarizeByClass(dataset)
    #print("Summary by class value{0}").format(summary)
    
    #x = 71.5
    #mean = 73
    #stdev = 6.2
    #probability = calculateProbability(x,mean,stdev)
    #print('Probability of belonging to this class: {0}').format(probability)
    
    #summaries = {0:[(1,0.5)],1:[(20,5.0)]}
    #inputVector = [1.1,'?']
    #probabilities = calculateClassProbabilities(summaries,inputVector) 
    #print('Probabilities for each class: {0}').format(probabilities)
    
    #summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
    #inputVector = [1.1, '?']
    #result = predict(summaries, inputVector)
    #print('Prediction: {0}').format(result)
   
    #testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
    #predictions = ['a', 'a', 'a']
    #accuracy = getAccuracy(testSet, predictions)
    #print('Accuracy: {0}').format(accuracy)
   
    filename = "pima-indians-diabetes.data.csv"
    splitRatio = 0.67
    dataset = LoadCsv(filename)
    trainingSet,testSet = splitDataset(dataset,splitRatio)
    print("Split {0} rows into train={1} and test = {2} rows").format(len(dataset),len(trainingSet),len(testSet))
    #Prepare model
    summaries = summarizeByClass(trainingSet)
    #test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%').format(accuracy)


if __name__ == "__main__":
    main()
    