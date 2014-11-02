import sys
sys.path.append('libsvm/python/')
sys.path.append('libsvm')
from svmutil import *
import numpy
import scipy.sparse
import scipy.sparse.linalg
import numpy.linalg
import re
from collections import defaultdict
from random import shuffle
from tfidf import tfidf
from classes import get_classes
from lsi import get_numfeats, lsi

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def get_acc(trainfilename, testfilename, d, a, r):
  features = lsi(tfidf('small_train.txt'), d)
  classes = get_classes('small_train.txt', a, r)
  prob  = svm_problem(classes, features)
  param = svm_parameter('-t 0 -c 4 -b 1')
  m = svm_train(prob, param)
  test_features = lsi(tfidf('small_test.txt'),d)
  test_classes = get_classes('small_test.txt', a, r)
  p_label, p_acc, p_val = svm_predict(test_classes, test_features, m)
  return p_acc

def gridsearch(trainfilename, testfilename, minD, maxD, stepD, minA, maxA, stepA, minR, maxR, stepR, outfilename):
  outfile = open(outfilename, "w")
  for d in xrange(minD, maxD + stepD, stepD):
    for a in frange(minA, maxA + stepA, stepA):
      for r in xrange(minR, maxR + stepR, stepR):
        acc = get_acc(trainfilename, testfilename, d, a, r)
        outfile.write(str(d) + "," + str(a) + "," + str(r) + "," + str(acc))
        outfile.flush()

  outfile.close()

if __name__ == '__main__':
  gridsearch('small_train.txt', 'small_test.txt', 10, 1000, 10, 0.1, 0.9, 0.05, 1, 15, 1, 'gridresults.txt')


