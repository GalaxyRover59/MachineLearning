from math import log
import operator


def calcShannonEnt(dataSet):
    """
    calculate Shannon entropy of one dataset
    :param dataSet: a list of data
    :return: Shannon entropy based on labels of data
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for vec in dataSet:
        vecLabel = vec[-1]
        labelCounts[vecLabel] = labelCounts.get(vecLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCounts.keys():
        prop = labelCounts[key] / numEntries
        shannonEnt -= prop * log(prop, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    Get subset of data whose feature equal to specific value, and each vector removed this feature
    :param dataSet: raw dataset
    :param axis: the i-th feature
    :param value: specific value
    :return: reduced dataset
    """
    retDataSet = []
    for vec in dataSet:
        if vec[axis] == value:
            reducedVec = vec[:axis]
            reducedVec.extend(vec[axis + 1:])
            retDataSet.append(reducedVec)
    return retDataSet


def creatDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # calculate each feature
        featureList = [example[i] for example in dataSet]
        uniqueValue = set(featureList)
        newEntropy = 0.0
        for value in uniqueValue:
            subDateSet = splitDataSet(dataSet, i, value)
            prop = len(subDateSet) / float(len(dataSet))
            newEntropy += prop * calcShannonEnt(subDateSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature


def majorityCnt(classList):
    """
    Get name of majority category in classList
    """
    classCount = {}
    for clsname in classList:
        classCount[clsname] = classCount.get(clsname, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


myData, labels = creatDataSet()
trees = createTree(myData, labels)
print(trees)
