from numpy import *
import operator
import kNN
from os import listdir


def classify0(inX, dataset, labels, k):
    dataSetSize = dataset.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataset  # 在0轴方向重复inX (dataSetSize)次，再与dataset相减
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistIndicies = distance.argsort()  # 从小到大排序，输出相应的索引
    classCount = {}
    for i in range(k):  # 找出距离最小的k个点，统计对应的标签出现次数
        classLabel = labels[sortedDistIndicies[i]]
        classCount[classLabel] = classCount.get(classLabel, 0) + 1
    sortedClassCout = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCout[0][0]


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumstr = int(fileStr.split('_')[0])
        hwLabels.append(classNumstr)
        trainingMat[i, :] = kNN.img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumstr = int(fileStr.split('_')[0])
        vectorUnderTest = kNN.img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("classifier result: %d; real result: %d" % (classifierResult, classNumstr))
        if (classifierResult != classNumstr):
            errorCount += 1.0
    print("total errors: %d" % errorCount)
    print("error rate: %f" % (errorCount / float(mTest)))


handwritingClassTest()
