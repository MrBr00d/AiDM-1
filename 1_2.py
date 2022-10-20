#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold 
import math
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import mean_squared_error


# In[ ]:


# Loading the dataset
# Reading dataset
dataset = pd.read_table('ratings.dat', header = None, sep = '::', engine = 'python', usecols = [0,1,2], names = ('UserID','MovieID', 'Ratings'))

dataset.head()

# Are there missing values?
dataset.isna().sum()


# Keep in mind that there are “gaps” in numbering of users and items. (Dictionaries? Renumber everything? …)
# What is meant with this??


# In[ ]:


#test + optimization cell
import time
start = time.time()
#initialize params
n_iter = 0
rme_list = list()
n_splits = 5
Threshold = 0.001
KF = KFold(n_splits=5, random_state=123, shuffle=True)
#Kfold loop (5 datasets)
for Train, Test in KF.split(dataset):
    dataset_2_Train, dataset_2_Test = dataset.loc[Train], dataset.loc[Test]
    dataset_split = dataset_2_Train.pivot(
    index='UserID',
    columns='MovieID',
    values='Ratings'
    )
    dataset1 = dataset_split.sub(dataset_split.mean(axis = 1), axis=0)
    dataset_end = dataset1.sub(dataset1.mean(axis = 0), axis = 1)
    dataset_split = dataset_end
    M = dataset_split.to_numpy()
    d = 2
    n = dataset_split.shape[0]
    m = dataset_split.shape[1]
    a = dataset_split.stack().mean()
    U = np.empty([n,d])
    V = np.random.randn(d, m) # random numbers to increase chance of reachine global minimum
    U = np.random.randn(n, d)
    uv = np.matmul(U,V)
    
    RME = 1000
    halt = True
    n_iter += 1
    #iterate as long as change of RME is bigger than threshold
    while halt:
        RME_old = RME
        #update U matrix
        for r in range(U.shape[0]):
            numerator = 0
            denominator = 0

            for s in range(d):
                U_rk = U[r,:]
                U_rk = np.delete(U_rk, s, 0)
                V_kj = np.delete(V, s, 0)
                V_sj = V[s,:]
                P = np.matmul(U_rk, V_kj)
                m_rj = M[r,:]
                numerator = np.multiply(V_sj,(np.subtract(m_rj, P)))
                numerator = np.nansum(np.multiply(numerator, (np.isfinite(m_rj))))
                denominator = np.square(V_sj)
                denominator = np.nansum(np.multiply(denominator, (np.isfinite(m_rj))))
                new_var = np.divide(numerator, denominator)
                U[r,s] = new_var
        #update V matrix
        for r in range(d):
            for s in range(V.shape[1]):
                V_ks = V[:,s]
                V_ks = np.delete(V_ks, r, 0)
                U_ik = np.delete(U, r, 1)
                U_ir = U[:,r]
                P = np.matmul(U_ik, V_ks)
                m_is = M[:,s]
                numerator = np.multiply(U_ir, (np.subtract(m_is, P)))
                numerator = np.nansum(np.multiply(numerator, (np.isfinite(m_is))))
                denominator = np.square(U_ir)
                denominator = np.nansum(np.multiply(denominator, (np.isfinite(m_is))))
                newvar = np.divide(numerator, denominator)
                V[r,s] = newvar
        #matrix multiplication and comparison to original matrix M + calc MSE
        uv = np.matmul(U,V)
        dif_squared_0 = np.nan_to_num(np.square(np.subtract(uv, M)))
        dif_squared_total_sum = np.sum(dif_squared_0, axis = 0).sum()
        N_non_0 = np.count_nonzero(dif_squared_0)
        RME = np.divide(dif_squared_total_sum, N_non_0)
        RMSE = np.sqrt(RME)
        halt = np.abs(RMSE - RME_old) > Threshold
    print(f'fold {n_iter}; RMSE = {RMSE}')
    rme_list.append(RME)
print('Avg=', np.mean(rme_list))
rme_list_mean = np.mean(rme_list)
print(f'Average RMSE over {n_iter} folds = {rme_list_mean}')
ende = time.time()
print('Total runtime = ', (ende - start))

