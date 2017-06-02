# -*- coding: UTF-8 -*- 
import os
import numpy as np 
def zeroMean(dataMat):        
	meanVal = np.mean(dataMat, axis = 0) 
	newData = dataMat - meanVal  
	return newData,meanVal 

def eigValPct(eigVals,percentage):  
    sortArray = np.sort(eigVals) 
    sortArray = sortArray[-1::-1] 
    arraySum = np.sum(sortArray)  
    tmpSum = 0  
    num = 0  
    for i in sortArray:  
        tmpSum += i  
        num += 1  
        if tmpSum >= arraySum * percentage:  
            return num  


def pca(dataMat,percentage=0.95):
	meanRemoved, meanVals = zeroMean(dataMat) 
	covMat = np.cov(meanRemoved,rowvar=0)  
	eigVals, eigVects = np.linalg.eig(np.mat(covMat))  
	k = eigValPct(eigVals,percentage) 
	eigValInd = np.argsort(eigVals)  
	eigValInd = eigValInd[-1:-(k+1):-1] 
	redEigVects = eigVects[:,eigValInd] 
	lowDDataMat = meanRemoved * redEigVects 
	reconMat = (lowDDataMat * redEigVects.T) + meanVals 
	return lowDDataMat, reconMat

def readData(filename):
	with open(filename, 'r') as f:
		f.readline()
		stringArr = [line.strip().split('\t')[1:] for line in f.readlines()]
		dataArr = [map(float, line) for line in stringArr]
		return np.mat(dataArr).T
def format(value):
    return "%.5f" % value
		
def writeData(filename, feature):
	feature = np.array(feature)
	formatted = [[format(v) for v in r] for r in feature]
	with open(filename, 'w') as f:
		for i in range(feature.shape[0]):
			for j in range(feature.shape[1]):
				f.write(formatted[i][j])
				f.write('\t')
			f.write('\n')

def check(filename):
	with open(filename, 'r') as f:
		f.readline()
		types = [line.strip().split('\t')[1] for line in f.readlines()]
	classType = list()
	for type in types:
		if type not in classType:
			classType.append(type)
	return classType
	

if __name__ == '__main__':
	#filename = '../data/microarray.original.txt'
	#filename = '../data/finalData.txt'
	filename = '../data/processedData_0.95.txt'
	labelfile = '../data/E-TABM-185.sdrf.txt'
	classType = check(labelfile)
	print "The number of class is {}".format(len(classType))
	print classType
	feature = readData(filename)
	print "Finish reading the data. The number is {}. The dimensions of feature are {}".format(feature.shape[0], feature.shape[1])
	#filename = '../data/processedData_0.90_second.txt'
	#new_feature, _ = pca(feature, 0.90)
	#print "Feature is reduced to {} dimensions.".format(new_feature.shape[1])
	#writeData(filename, new_feature)
	#print "New feature has been written. The number is {}. The new dimensions are {}".format(new_feature.shape[0], new_feature.shape[1])
