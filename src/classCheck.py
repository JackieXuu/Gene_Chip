# -*- coding: UTF-8 -*- 
import os
import numpy as np
from collections import Counter
import pprint
pp = pprint.PrettyPrinter()


def read(filename):
	with open(filename, 'r') as f:
		f.readline()
		line = f.readline()
		print line.replace(" ", "").split('\t')

def check(filename):
	with open(filename, 'r') as f:
		f.readline()
		types = [line.strip() for line in f.readlines()]
	classType = Counter()
	for type in types:
		classType[type] += 1
	return classType


if __name__ == '__main__' :
	labelfile = '../data/disease.txt'
	classType = check(labelfile)
	print "The number of class is {}".format(len(classType))
	pp.pprint(classType)
