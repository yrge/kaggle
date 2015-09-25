__author__ = 'Stella Zhao'

import scipy.stats as stats
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import numpy as np
import os
from sklearn.linear_model import LinearRegression as LinearRegression, Ridge as Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import linear_model
from pylab import *
import matplotlib
#from dataclean import ReadOA, WriteOA
from combination1 import *
from combination1 import get_com_pare

path = u'//Users/zhaoshihuan/Desktop/pingan/'
path_result1 = path+u'result/nointer/'
path_result2 = path+u'result/inter/'


path_result = path_result1

def load_data(filename, ind):
    #data = np.loadtxt(filename)
    data = pd.read_csv(filename,sep = ' ',header = None)
    y = data.iloc[:, 1]
    lb = y
    #     lb[y > 0] = 1
    #     lb[y <= 0] = 0
    x = data.iloc[:, 2:]
    x = np.array(x)
    x_in = np.zeros((x.shape[0], ind.shape[0]), dtype=np.float)
    for i_var in range(ind.shape[0]):
        i = ind.iloc[i_var, 0] - 1
        j = ind.iloc[i_var, 1] - 1
        if j == -1:  # single variable
            x_in[:, i_var] = x[:, i]
        else:
            x_in[:, i_var] = x[:, i] * x[:, j]
    return x_in, lb




def PredictorsColumn(filename):
    result = pd.DataFrame(columns=['p1', 'p2'])
    coef = pd.read_csv(filename, header=None)
    number0fCoef = int(coef.ix[0, 0])
    p = int(coef.ix[number0fCoef + 2, 0])
    columns = coef.ix[2:number0fCoef + 1, 0]
    for i in columns:
        result = result.append(PredictorsUsedInColumn(int(i.split()[0]), p))
    return result

def PredictorsUsedInColumn(index, p):
    p1 = p2 = 0
    if (index <= p):  # simple predictor
        p1 = index
        p2 = -1
    else:
        index -= p;
        while (index + p2 > p):  # so that index + *x2 <= p when break
            index -= p - p2
            p2 += 1

        p1 = index + p2
        p2 += 1
    return pd.DataFrame({'p1': [p1], 'p2': [p2]})

def fit_linear_model(X_parameters, Y_parameters,Xt):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)

    y_pred = regr.predict(X_parameters)
    yt_pred = regr.predict(Xt)
    return y_pred,yt_pred



def WriteOA(pdobj,filename,data_path=path_result,index=False,header=True,sep=',',encoding='gbk'):
    pdobj.to_csv(data_path+filename, index=index, header=header, sep=sep, encoding=encoding)


def load_data_easy1(path):  # loaddata
    data = pd.read_csv(path,sep=' ',header=None)
    data = data.T
    y = data.iloc[:,-1]
    del data[data.shape[1]-1]
    data.insert(0,'y',y)
    return data

def load_data_easy(path):  # loaddata
    data = np.loadtxt(path)
    data = data.T
    return data


def sep_random(data, ratio):  # ratio is train%
    items = range(data.shape[0])
    random.shuffle(items)
    N = data.shape[0] * ratio
    N = int(N)
    items_train = items[0:N]
    items_test = items[N:data.shape[0]]
    train = data.iloc[items_train,]
    test = data.iloc[items_test,]
    return train, test


def pre_model2():  # 训练集与测试集已经分好
    ratio = 0.8
    weight = 10
    train = load_data_easy(path+'train.csv')
    train.rename(columns ={0:'y'},inplace = True)
    test = load_data_easy(path+'test.csv')
    test.rename(columns ={0:'y'},inplace = True)
    #frames = [data,data1]
    #data = pd.concat(frames)

    train.insert(0, 'weight', 1)
    test.insert(0, 'weight', 1)

    train['weight'] = (train['y']==1).map(lambda x:weight if x else 1)
    test['weight'] = (train['y']==1).map(lambda x:weight if x else 1)

    #to_csv(path+'/all.csv',sep=' ',index=False, header=False)

    #检查训练集和测试集中1的比例
    #train, test = sep_random(data, ratio)
    bbbb = test.iloc[:,1]
    rate0 = 1-bbbb.sum()*1.0/len(bbbb)
    print rate0
    bbbb = train.iloc[:,1]
    rate0 = 1-bbbb.sum()*1.0/len(bbbb)
    print rate0

    train.to_csv(path_result + 'est.dat', sep=' ', index=False, header=False)
    test.to_csv(path_result + 'validation.data', sep=' ', index=False, header=False)


def pre_model1():  # 训练集与测试比例,可以调
    ratio = 0.8
    data = load_data_easy(path+'train.csv')
    data1 = load_data_easy(path+'test.csv')
    future = load_data_easy(path+'future.csv')

    frames = [data,data1]
    data = pd.concat(frames)
    data.rename(columns ={0:'y'},inplace = True)
    data.insert(0, 'weight', 1)
    future.insert(0, 'weight', 1)

    data['weight'] = (data['y']==1).map(lambda x:10 if x else 1)
    data.to_csv(path+'all.csv',sep=' ',index=None, header=None)

    #检查训练集和测试集中1的比例
    train, test = sep_random(data, ratio)
    bbbb = test.iloc[:,1]
    rate0 = 1-bbbb.sum()*1.0/len(bbbb)
    print rate0
    bbbb = train.iloc[:,1]
    rate0 = 1-bbbb.sum()*1.0/len(bbbb)
    print rate0
    bbbb = future.iloc[:,1]
    rate0 = 1-bbbb.sum()*1.0/len(bbbb)
    print rate0



    train.to_csv(path_result + 'est.dat', sep=' ', index=False, header=False)
    test.to_csv(path_result + 'validation.data', sep=' ', index=False, header=False)
    future.to_csv(path_result + 'future.data', sep=' ', index=False, header=False)

def pre_model4():  # 训练集与测试比例,可以调，用ReadOA读取数据,这里可以看出日均150比以上的atm很少
    ratio = 0.8
    #data = ReadOA('sw_res_eff_beijing.txt')
    data = ReadOA('sw_res_eff_fourcity.txt')

    del data['city_shanghai']
    data.insert(0, 'weight', 1)
    data.rename(columns ={'avg_monthly':'y'},inplace = True)


    #data['weight'] = (data['y']==1).map(lambda x:10 if x else 1)

    #检查训练集和测试集中1的比例
    train, test = sep_random(data, ratio)
    bbbb = test.iloc[:,1]
    bbbb = trans0(bbbb,4500)
    rate1 = bbbb.sum()*1.0/len(bbbb)
    print rate1
    bbbb = train.iloc[:,1]
    bbbb = trans0(bbbb,4500)
    rate1 = bbbb.sum()*1.0/len(bbbb)
    print rate1

    train1 = train
    test1 = test
    train1['weight'] = (train['y']>4500).map(lambda x:20 if x else 1) #exp1.2
    #train['weight'] = (data['y']>4500).map(lambda x:20 if x else 1) #exp1.3
    #train['weight'] = 1#exp1

    train1['y'] = (train['y']>4500).map(lambda x:1 if x else 0)
    test1['y'] = (test['y']>4500).map(lambda x:1 if x else 0)

    names = train.columns.values
    names = pd.DataFrame(names)



    train1.to_csv(path_result + 'est.dat', sep=' ', index=False, header=False)
    test1.to_csv(path_result + 'validation.data', sep=' ', index=False, header=False)
    WriteOA(names,'names')


def ReadOA(filename, data_path=path, encoding='utf-8', sep=' ', skipinitialspace=False):
    oa = pd.read_csv(data_path + filename, encoding=encoding, sep=sep, skipinitialspace=skipinitialspace)
    return oa

def trans0(y,thresh):
    trans = {-1:0}
    y= np.sign(y -thresh)
    y = y.astype(np.int32)
    y = pd.DataFrame(y)
    y = y.replace(trans)
    y = np.array(y)
    return y

def fit_data1(lb_pred,lb_t_pred,i,result,X,lb,Xt,lb_t,claf_name,thresh):#计算0/1的分类结果
    #train, test = sep_random(data, ratio)
    #for i in range(iter):
    #claf.fit(X,lb)

    lb_pred = trans0(lb_pred,thresh)
    lb_t_pred =trans0(lb_t_pred,thresh)


    lb= trans0(lb,thresh)
    lb_t= trans0(lb_t,thresh)

    y_pred = lb_pred
    result =pd.DataFrame(result)

    result.ix[i,'estimator'] = claf_name
    result.ix[i, 'train_pre'] = (lb == y_pred).sum()*1.0/len(lb)

    result.ix[i, 'call_0_train'] = ((lb == y_pred) & (lb == 0)).sum() * 1.0 / (lb == 0).sum()
    result.ix[i, 'call_1_train'] = ((lb == y_pred) & (lb == 1)).sum() * 1.0 / (lb == 1).sum()
    result.ix[i, 'pre_0_train'] = ((lb == y_pred) & (y_pred == 0)).sum() * 1.0 / (y_pred == 0).sum()
    result.ix[i, 'pre_1_train'] = ((lb == y_pred) & (y_pred == 1)).sum() * 1.0 / (y_pred == 1).sum()
    result.ix[i, 'recommend_train'] = 1-(y_pred == 1).sum() * 1.0 / y_pred.shape[0]
    result.ix[i, '1rate_train'] = (lb == 1).sum() * 1.0 / lb.shape[0]

    yt_pred = []
    yt_pred = lb_t_pred

    result.ix[i, 'test_pre'] = (lb_t == yt_pred).sum()*1.0/len(lb_t)
    result.ix[i, 'call_0_test'] = ((lb_t == yt_pred) & (lb_t == 0)).sum() * 1.0 / (lb_t == 0).sum()
    result.ix[i, 'call_1_test'] = ((lb_t == yt_pred) & (lb_t == 1)).sum() * 1.0 / (lb_t == 1).sum()
    result.ix[i, 'pre_0_test'] = ((lb_t == yt_pred) & (yt_pred == 0)).sum() * 1.0 / (yt_pred == 0).sum()
    result.ix[i, 'pre_1_test'] = ((lb_t == yt_pred) & (yt_pred == 1)).sum() * 1.0 / (yt_pred == 1).sum()
    result.ix[i, 'recommend_test'] = 1-(yt_pred == 1).sum() * 1.0 / yt_pred.shape[0]
    result.ix[i, '1rate_test'] = (lb_t == 1).sum() * 1.0 / lb_t.shape[0]
    return result



if __name__=='__main__':
    
    test()


def test():

    pre_model4()

    print 'wait for the sweeping results'



    for i in range(1,9):

        model = 'model.00'+str(i)

        print i
        for thresh in range(3000,3010):
            print thresh
            res = comput(model,thresh)
            print res.call_1_test
            print res.call_0_test

    thresh = 4500.0
    for i in range(10,29):
        model = 'model.0'+str(i)
        print 'num of var:'+ str(i)
        res = comput(model,thresh)
        print res.pre_1_test
        print res.pre_0_test



def comput_future(model):   #计算future
    ind = PredictorsColumn(path_result + model)
    estimators = {}
    alpha = 50
    estimators['decision_tree'] = tree.DecisionTreeClassifier(max_depth=5)
    #estimators['svm_c_linear'] = svm.SVC(kernel='linear')
    estimators['svm_linear'] = svm.LinearSVC()
    estimators['logister_regression_l1'] = linear_model.LogisticRegression(penalty='l1', C=alpha, tol=1e-10)
    estimators['logister_regression_l2'] = linear_model.LogisticRegression(penalty='l2', C=alpha, tol=1e-10)
    estimators['random_forest'] = RandomForestClassifier(max_depth=5, criterion='gini', n_estimators=2000, n_jobs=5)
    result = pd.DataFrame({'estimator':'','call_0_train': [0], 'pre_0_train': [0], 'call_1_train': [0], 'pre_1_train': [0],
                           'recommend_train': [0], '1rate_train': [0], 'train_pre': [0],
                           'call_0_test': [0], 'pre_0_test': [0], 'call_1_test': [0], 'pre_1_test': [0],
                           'recommend_test': [0], '1rate_test': [0], 'test_pre': [0]})


    X, lb = load_data(path_result + 'est.dat', ind)
    Xt, lb_t = load_data(path_result + 'future.data', ind)



    i = 0
    for claf_name in estimators.keys():
        claf = estimators[claf_name]

        print claf_name
        claf.fit(X,lb)

        lb_pred = claf.predict(X)
        save = pd.DataFrame(lb)
        save.insert(0,'ypred',lb_pred)
        save.to_csv(path_result+claf_name+'_pre_train.csv')

        lb_t_pred = claf.predict(Xt)
        save = pd.DataFrame(lb_t)
        save.insert(0,'ypred',lb_t_pred)
        save.to_csv(path_result+claf_name+'_pre_test.csv')

        result = fit_data1(lb_pred,lb_t_pred,i,result,X,lb,Xt,lb_t,claf_name)
        i = i+1

    lb_pred ,lb_t_pred = fit_linear_model(X,lb,Xt)
    result = fit_data1(lb_pred,lb_t_pred,i,result,X,lb,Xt,lb_t,'linear_model')

    aa = result['estimator']
    del result['estimator']
    result.insert(0, 'estimator', aa)
    return result



def comput(model,thresh):
    ind = PredictorsColumn(path_result + model)
    estimators = {}
    alpha = 50
    estimators['decision_tree'] = tree.DecisionTreeClassifier(max_depth=5)
    #estimators['svm_c_linear'] = svm.SVC(kernel='linear')
    estimators['svm_linear'] = svm.LinearSVC()
    estimators['logister_regression_l1'] = linear_model.LogisticRegression(penalty='l1', C=alpha, tol=1e-10)
    estimators['logister_regression_l2'] = linear_model.LogisticRegression(penalty='l2', C=alpha, tol=1e-10)
    estimators['random_forest'] = RandomForestClassifier(max_depth=5, criterion='gini', n_estimators=2000, n_jobs=5)
    result = pd.DataFrame({'estimator':'','call_0_train': [0], 'pre_0_train': [0], 'call_1_train': [0], 'pre_1_train': [0],
                           'recommend_train': [0], '1rate_train': [0], 'train_pre': [0],
                           'call_0_test': [0], 'pre_0_test': [0], 'call_1_test': [0], 'pre_1_test': [0],
                           'recommend_test': [0], '1rate_test': [0], 'test_pre': [0]})


    X, lb = load_data(path_result + 'est.dat', ind)
    Xt, lb_t = load_data(path_result + 'validation.data', ind)



    i = 0
    for claf_name in estimators.keys():
        claf = estimators[claf_name]

        print claf_name
        claf.fit(X,lb)

        lb_pred = claf.predict(X)
        save = pd.DataFrame(lb)
        save.insert(0,'ypred',lb_pred)
        save.to_csv(path_result+claf_name+'_pre_train.csv')

        lb_t_pred = claf.predict(Xt)
        save = pd.DataFrame(lb_t)
        save.insert(0,'ypred',lb_t_pred)
        save.to_csv(path_result+claf_name+'_pre_test.csv')

        result = fit_data1(lb_pred,lb_t_pred,i,result,X,lb,Xt,lb_t,claf_name,thresh)
        i = i+1

    lb_pred ,lb_t_pred = fit_linear_model(X,lb,Xt)
    result = fit_data1(lb_pred,lb_t_pred,i,result,X,lb,Xt,lb_t,'linear_model',thresh)

    aa = result['estimator']
    del result['estimator']
    result.insert(0, 'estimator', aa)
    return result
    #result.to_csv(path_result+'inter.csv')


def comput3():#不选变量

    train = pd.read_csv(path_result+'est.dat',sep = ' ',header = None)
    test = pd.read_csv(path_result+'validation.data',sep = ' ',header = None)
    #future = pd.read_csv(path_result+'future.data',sep = ' ',header = None)

    estimators = {}
    alpha = 50
    estimators['decision_tree'] = tree.DecisionTreeClassifier(max_depth=5)
    #estimators['svm_c_linear'] = svm.SVC(kernel='linear')
    estimators['svm_linear'] = svm.LinearSVC()
    estimators['logister_regression_l1'] = linear_model.LogisticRegression(penalty='l1', C=alpha, tol=1e-10)
    estimators['logister_regression_l2'] = linear_model.LogisticRegression(penalty='l2', C=alpha, tol=1e-10)
    estimators['random_forest'] = RandomForestClassifier(max_depth=5, criterion='gini', n_estimators=2000, n_jobs=5)
    result = pd.DataFrame({'estimator':'','call_0_train': [0], 'pre_0_train': [0], 'call_1_train': [0], 'pre_1_train': [0],
                           'recommend_train': [0], '1rate_train': [0], 'train_pre': [0],
                           'call_0_test': [0], 'pre_0_test': [0], 'call_1_test': [0], 'pre_1_test': [0],
                           'recommend_test': [0], '1rate_test': [0], 'test_pre': [0]})


    lb = train[1].values
    X = train.iloc[:,2:]
    #lb = load_data(path_result + 'est.dat', ind)
    Xt = test.iloc[:,2:]
    lb_t = test[1].values



    i = 0
    for claf_name in estimators.keys():
        claf = estimators[claf_name]

        print claf_name
        claf.fit(X,lb)

        lb_pred = claf.predict(X)
        save = pd.DataFrame(lb)
        save.insert(0,'ypred',lb_pred)
        save.to_csv(path_result+claf_name+'_pre_train.csv')

        lb_t_pred = claf.predict(Xt)
        save = pd.DataFrame(lb_t)
        save.insert(0,'ypred',lb_t_pred)
        save.to_csv(path_result+claf_name+'_pre_test.csv')

        result = fit_data1(lb_pred,lb_t_pred,i,result,X,lb,Xt,lb_t,claf_name)
        i = i+1

    lb_pred ,lb_t_pred = fit_linear_model(X,lb,Xt)
    result = fit_data1(lb_pred,lb_t_pred,i,result,X,lb,Xt,lb_t,'linear_model')

    aa = result['estimator']
    del result['estimator']
    result.insert(0, 'estimator', aa)
    return result

def pre_model5():  # 训练集与测试比例,可以调,提前组合变量
    #ratio = 0.8
    #data = np.loadtxt(path+'train.csv')


    #data=load_data_easy(path+'train.csv')
    #data = pd.DataFrame(data)
    #data = data.T
    #data.insert(0, 'weight', 1)


    train = load_data_easy(path+'train.csv')
    test = load_data_easy(path+'test.csv')
    train1,test1 = get_com_pare(train,test)

    #future = load_data_easy(path+'future.csv')


    #data = pd.concat(frames)
    train1.rename(columns ={0:'y'},inplace = True)
    test1.rename(columns ={0:'y'},inplace = True)
    train1.insert(0, 'weight', 1)
    test1.insert(0, 'weight', 1)
    #future.insert(0, 'weight', 1)

    train1['weight'] = (train1['y']==1).map(lambda x:8 if x else 1)
    #train1['weight'] = (data['y']==1).map(lambda x:10 if x else 1)
    #data.to_csv(path+'all.csv',sep=' ',index=None, header=None)

    #检查训练集和测试集中1的比例
    #train, test = sep_random(data, ratio)
    bbbb = test1['y']
    rate0 = 1-bbbb.sum()*1.0/len(bbbb)
    print rate0
    bbbb = train1.iloc[:,1]
    rate0 = 1-bbbb.sum()*1.0/len(bbbb)
    print rate0
    #bbbb = future.iloc[:,1]
    #rate0 = 1-bbbb.sum()*1.0/len(bbbb)
    #print rate0
    train.to_csv(path_result + 'est.dat', sep=' ', index=False, header=False)
    test.to_csv(path_result + 'validation.data', sep=' ', index=False, header=False)
    #future.to_csv(path_result + 'future.data', sep=' ', index=False, header=False)

