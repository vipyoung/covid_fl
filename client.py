# -*- coding: utf-8 -*-

import sys
import socket
import time
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.utils import shuffle

import encdec

def recv(soc, buffer_size = 4096, recv_timeout = 10):
    received_data = b""
    while True:
        try:
            soc.settimeout(recv_timeout)
            received_data += soc.recv(buffer_size)
        except socket.timeout:
            print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds".format(recv_timeout=recv_timeout))
            return None, 0
        except BaseException as e:
            return None, 0
            print("An error occurred while receiving data from the server {msg}".format(msg=e))
        if received_data.decode('utf-8')[-1] == '\4': # end of transmission
            break
    try:
        received_data = received_data.decode('utf-8')[:-1]
        received_data = json.loads(received_data, cls = encdec.MessageDecoder)
    except BaseException as e:
        print("Error Decoding the Client's Data: {msg}\n".format(msg=e))
        return None, 0
    return received_data, 1

server_ip = 'localhost'
server_port = 10000

#----- INITIALIZATION

bim = pd.read_csv('./MergeM2.bim', header = None, sep = '\t')
ped = pd.read_csv('./MergeM2.ped', header = None, index_col = 0, sep = '\t')
ped.drop(ped.columns[range(5)], axis = 1, inplace = True)
ref_ref = (bim[5] + ' ' + bim[5]).values.reshape((1,-1)) 
feature_genes = pd.DataFrame((ped.values != ref_ref).astype(int), columns = bim[1], index = ped.index) # this dataframe has samples along the rows and genes along the columns, features_local[i,j] is 1 if gene j in sample i is mutated
pheno = pd.read_csv('./pheno.txt', sep = ' ', index_col = 0)
covar = pd.read_csv('./covar.txt', sep = ' ', index_col = 0)
index_samples = set(feature_genes.index) & set(pheno.index) & set(covar.index)
phenotypes = pheno.loc[index_samples]['A2'] # where A2 is the name of the column of the pheno.txt considered for the analysis
age_max = 100
covar = covar.loc[index_samples][['age', 'sex']]
covar['age'][covar['age'] > age_max] = age_max
covar['age'] = covar['age'] / age_max
covar['age*sex'] = covar['age']*covar['sex']
covar['age^2'] = covar['age']**2
covar['age^2*sex'] = covar['age']**2*covar['sex']
X = np.concatenate((covar.values, feature_genes.loc[index_samples].values), axis = 1) # np.array of input features
y = phenotypes.values # np.array of output classes
feature_names = np.array(['age','sex','age*sex','age^2','age^2*sex'] + list(feature_genes.columns))
print('Number of samples: {}'.format(X.shape[0]))
print('Number of features: {}'.format(X.shape[1]), len(feature_names))
print('Samples per phenotype class: {}'.format(', '.join(['{} = {}'.format(class_id, n_samples) for class_id, n_samples in zip(*np.unique(y, return_counts = True))])))

#--- dummy
#from sklearn.preprocessing import PolynomialFeatures
#n_samples = 1000
#n_samples_half = int(0.5*n_samples)
#x0 = np.random.randint(low = 0, high = 2, size = n_samples)
#x1 = np.random.randint(low = 0, high = 2, size = n_samples)
#x2 = np.random.randint(low = 0, high = 2, size = n_samples)
#x3 = np.random.randint(low = 0, high = 2, size = n_samples)
#noise_term = (np.abs(np.random.normal(loc = 0.0, scale = 1, size = n_samples) ) > 2.0)
#pred_01 = (np.logical_xor(x0,x1).astype(int) + noise_term) % 2
#pred_23 = (np.logical_xor(x2,x3).astype(int) + noise_term) % 2
#y = np.empty(n_samples)
#if len(sys.argv) == 1: # combination of two rules
#    print('# Combination of two rules')
#    y[:n_samples_half] = pred_01[:n_samples_half]
#    y[n_samples_half:] = pred_23[n_samples_half:]
#    X = np.concatenate((x0.reshape(-1,1),x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1)), axis = 1)
#    poly = PolynomialFeatures(3, include_bias = False)
#    X = poly.fit_transform(X)
#    for i_noise in range(1000):
#        noise_term =  np.random.randint(low = 0, high = 2, size = (n_samples,1))
#        X = np.concatenate((X, noise_term), axis = 1)
#    feature_names = ['A'+str(i) for i in range(X.shape[1])]
#elif len(sys.argv) == 2: # using alternatively one or the other rule
#    if int(sys.argv[1]) % 2  == 0:
#        print('# Using rule 0')
#        y = pred_01
#        X = np.concatenate((x0.reshape(-1,1),x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1)), axis = 1)
#        poly = PolynomialFeatures(3, include_bias = False)
#        X = poly.fit_transform(X)
#        for i_noise in range(1000):
#            noise_term =  np.random.randint(low = 0, high = 2, size = (n_samples,1))
#            X = np.concatenate((X, noise_term), axis = 1)
#        feature_names = ['A'+str(i) for i in range(50)] +  ['B'+str(i) for i in range(50, X.shape[1])]
#    else:
#        print('# Using rule 1')
#        y = pred_23
#        X = np.concatenate((x0.reshape(-1,1),x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1)), axis = 1)
#        poly = PolynomialFeatures(3, include_bias = False)
#        X = poly.fit_transform(X)
#        for i_noise in range(750):
#            noise_term =  np.random.randint(low = 0, high = 2, size = (n_samples,1))
#            X = np.concatenate((X, noise_term), axis = 1)
#        feature_names = ['A'+str(i) for i in range(50)] +  ['C'+str(i) for i in range(50, X.shape[1])]
#X, y = shuffle(X, y)
#--- END: dummy

#----- END: INITIALIZATION

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
try:
    soc.connect((server_ip, server_port))
except BaseException as e:
    print('Error Connecting to the Server: {msg}'.format(msg=e))
    soc.close()
    print('Socket Closed')
print('Successful Connection to the Server\n')

while True:
    print("Receiving from the Server")
    received_message, status = recv(soc=soc, buffer_size = 4096, recv_timeout = 10)
    if status == 0:
        break
    #print('DEBUG',received_message, end="\n\n")
    response_message = {}
    if received_message['what2do'] == 'split':
        skf = StratifiedKFold(n_splits = received_message['k_folds'])
        train_indexes, test_indexes = [], []
        for train_index, test_index in skf.split(X, y):
            train_indexes.append(train_index)
            test_indexes.append(test_index)
        response_message['what2do'] = 'ready'
    elif received_message['what2do'] == 'train_all_samples':
        train_indexes, test_indexes = [range(X.shape[0]),], [range(X.shape[0]),] # these are not really used as training/testing set, it's just for consistency with the 'minimize' section in cross-validation steps
        response_message['what2do'] = 'ready'
    elif received_message['what2do'] == 'minimize':
        model = received_message['model']
        if isinstance(received_message['coef_'], pd.Series):
            model.coef_ = received_message['coef_'].loc[feature_names].values
            model.intercept_ = np.array([model.intercept_,])
        i_fold = received_message['i_fold']
        if i_fold >= len(train_indexes):
            response_message['what2do'] == 'echo'
        else:
            train_index = train_indexes[i_fold]
            test_index = test_indexes[i_fold]
            model.partial_fit(X[train_index,:], y[train_index], classes = [0,1])
            print('---logloss', log_loss(y[train_index], model.predict_proba(X[train_index,:]), eps=1e-15) )
            print('---top features:',np.argsort(np.abs(model.coef_.flatten()))[-4::])
            response_message['what2do'] = 'update'
            response_message['n_samples'] = len(train_index)
            response_message['confusion_matrix'] = confusion_matrix(y[test_index], model.predict(X[test_index,:]))
            response_message['model'] = model
            response_message['coef_'] = pd.Series(model.coef_.flatten(), index = feature_names)
    elif received_message['what2do'] == 'echo':
        response_message = received_message
    elif received_message['what2do'] == 'reset':
        response_message = received_message
    elif received_message['what2do'] == 'done':
        print('DONE')
        break
    else:
        print('Unrecognized message type')
        break

    response_json = json.dumps(response_message, cls = encdec.MessageEncoder) + '\4'
    print('Sending Data to the Server')
    response_bin = bytes(response_json, encoding="utf-8")
    soc.sendall(response_bin)

soc.close()
print('Socket Closed\n')
