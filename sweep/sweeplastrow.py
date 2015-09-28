__author__ = 'zhaoshihuan'
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import numpy as np
import os
import matplotlib
#from dataclean import ReadOA, WriteOA




#path = u'//Users/zhaoshihuan/Desktop/pingan/'

#incrSweepMat = np.array([1.2,2,34,2.98])

#incrSweepMat = np.matrix(incrSweepMat)
#incrSweepMat = incrSweepMat.T


#columnIndex = incrSweepMat[:,0]

#j = columnIndex[newRow]
#CPj = incrSweepMat[:,1:]

def SweepLastRow(sm):#sm 的type为np.matrix
    newRow = sm.newRow - 1
    columnIndex = sm.incrSweepMat[:,0]#diffrient with cpp,the first col is index_col
    sm.skipVector.ix[0,1] = 1

    j = columnIndex[newRow]
    j = int(j)
    CPj = sm.incrSweepMat[newRow,1:]

    #step 1:sweep previous row from new row
    for row in range(1,newRow):
        if k ==-1:break
        CPk = sm.incrSweepMat[newRow,1:]
        CPj = SweepOut(k,CPk,j,CPj)
        sm.incrSweepMat[newRow,1:] = SweepOut(k,CPk,j,CPj)


    #step 2:ToPivotForm(),incrSweepMat finish transforming
    CPj = ToPivotForm(j,CPj)
    sm.incrSweepMat[newRow,1:] = CPj
    sm.fullSweepMat[newRow,1:] = CPj


    #step 3:AjustRSS()
    sm.residualSS = AjustRSS(j,CPj)

    #step 5:sweep new row from y in incremental
    sm.incrSweepMat[0,1:] = SweepOut(j,CPj,0,sm.incrSweepMat[0,1:])

    #step 6:sweep new row from previous row in fully sweep
    for row in range(1,newRow):
        k = columnIndex[row]
        CPk = sm.fullSweepMat[row,1:]
        sm.fullSweepMat[k,1:] = SweepOut(j,CPj,k,CPk)






def AjustRSS(j,CPj):
    pivot = CPj[0,j]
    residualSS1 = sm.residualSS
    skipVector = sm.skipVector
    #skipVector为1表示已选入模型为－1表示未选入模型
    residualSS1[0,0] += residualSS1[0,0]+np.square(CPj[0,0])/pivot
    for i in range(1,sm.NumberOfVariable-1):
        residualSS1[0,i] += -skipVector.ix[i,1]*np.square(CPj[0,i])/pivot
    residualSS1[0,j] = - pivot
    return residualSS





def ToPivotForm(j,CPj):
    j = int(j)

    vjj = CPj[0,j]
    for i in range(CPj.shape[1]):
        CPj[0,i] /= vjj
    CPj[0,j] = - 1.0/ vjj
    return CPj

def SweepOut(k,CPk,j,CPj):
    vjk = CPj[0,k]
    for i in range(CPj.shape[1]):
        CPj[0,i] += -  vjk*CPk[0,i]
    CPj[0,k] = -  vjk*CPk[0,k]
    return CPj



class sm(object):
    def __init__(path):
        train = pd.ReadOA
        self =
    newRow = 2
    NumberOfVariable = 3

    aa = np.zeros([NumberOfVariable,NumberOfVariable+1])
    bb = range(1,NumberOfVariable)
    skipVector = pd.DataFrame({'col':bb,'iskip':np.zeros(NumberOfVariable-1)-1})

    incrSweepMat = np.matrix(aa)
    fullSweepMat = np.matrix(aa)
    #run-test

    incrSweepMat[0,:] = [0,26196.2095,12730.5881,587.0357]
    incrSweepMat[1,:] = [1,12730.58881,6934.3324,282.3329]


    fullSweepMat[0,:] = [0,26196.2095,12730.5881,587.0357]
    fullSweepMat[1,:] = [1,12730.58881,6934.3324,282.3329]

    residualSS = incrSweepMat[0,1:]


class Student(object):

    def __init__(self, name, score):
        self.__name = name
        self.__score = score

    def print_score(self):
        print '%s: %s' % (self.__name, self.__score)

    def set_score(self, score):
        if 0 <= score <= 100:
            self.__score = score
        else:
            raise ValueError('bad score')