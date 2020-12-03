import socket
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.utils import shuffle

def recv(soc, buffer_size=1048576, recv_timeout=10):
    received_data = b""
    while str(received_data)[-2] != '.':
        try:
            soc.settimeout(recv_timeout)
            received_data += soc.recv(buffer_size)
        except socket.timeout:
            print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds.".format(recv_timeout=recv_timeout))
            return None, 0
        except BaseException as e:
            return None, 0
            print("An error occurred while receiving data from the server {msg}.".format(msg=e))

    try:
        received_data = pickle.loads(received_data)
    except BaseException as e:
        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
        return None, 0

    return received_data, 1

server_ip = 'localhost'
server_port = 10000

#----- INITIALIZATION
bim = pd.read_csv('./MergeM1.bim', header = None, sep = '\t')
ped = pd.read_csv('./MergeM1.ped', header = None, index_col = 0, sep = '\t')
ped.drop(ped.columns[range(5)], axis = 1, inplace = True)
ref_ref = (bim[5] + ' ' + bim[5]).values.reshape((1,-1)) 
features_local = pd.DataFrame((ped.values != ref_ref).astype(int), columns = bim[1], index = ped.index) # this dataframe has samples along the rows and genes along the columns, features_local[i,j] is 1 if gene j in sample i is mutated
with open('genes_list.txt','rt') as fin:
    genes_list = [l.strip() for l in fin.readlines() if l.strip()]
features = pd.DataFrame(np.zeros((len(ped.index), len(genes_list))).astype(int), index = ped.index, columns = genes_list)
for gene in features_local.columns:
    features[gene] = features_local[gene]
pheno = pd.read_csv('./pheno.txt', sep = '\t', index_col = 0)
phenotypes = pheno.loc[features.index]['A1'] # where A1 is the name of the column of the pheno.txt considered for the analysis
X = features.values # np.array of input features
y = phenotypes.values # np.array of output classes
#----- END: INITIALIZATION

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print('Socket Created\n')
try:
    soc.connect((server_ip, server_port))
except BaseException as e:
    print('Error Connecting to the Server: {msg}'.format(msg=e))
    soc.close()
    print('Socket Closed')
print('Successful Connection to the Server\n')

while True:
    print("Receiving from the Server")
    received_message, status = recv(soc=soc, buffer_size=1048576, recv_timeout=10)
    if status == 0:
        break
    print(received_message, end="\n\n")
    response_message = {}
    if received_message['what2do'] == 'split':
        skf = StratifiedKFold(n_splits = received_message['k_folds'])
        train_indexes, test_indexes = [], []
        for train_index, test_index in skf.split(X, y):
            train_indexes.append(train_index)
            test_indexes.append(test_index)
        response_message['what2do'] = 'ready'
    elif received_message['what2do'] == 'minimize':
        model = received_message['model']
        i_fold = received_message['i_fold']
        if i_fold >= len(train_indexes):
            response_message['what2do'] == 'echo'
        else:
            train_index = train_indexes[i_fold]
            test_index = test_indexes[i_fold]
            model.fit(X[train_index,:], y[train_index])
            print('---logloss', log_loss(y[train_index], model.predict_proba(X[train_index,:]), eps=1e-15) )
            response_message['what2do'] = 'update'
            response_message['model'] = model
            response_message['confusion_matrix'] = confusion_matrix(y[test_index], model.predict(X[test_index,:]))
            response_message['n_samples'] = len(train_index)
            #time.sleep(1)
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

    response_data = pickle.dumps(response_message)
    print('Sending Data to the Server\n')
    print(response_message, end="\n\n")
    soc.sendall(response_data)

soc.close()
print('Socket Closed\n')


"""
# comparison between the solution of the federated learning and the standard one
import matplotlib.pyplot as pl
import pandas as pd

def make_plot(model):
    df_res = pd.DataFrame(np.arange(X.shape[1]))
    df_res['Feature Importance'] = model.coef_[0]
    df_res['abs_score'] = np.abs(model.coef_[0])
    df_plot = df_res.sort_values(['abs_score'], ascending=False)
    df_plot[df_plot['abs_score']>0].plot.bar(x=0, y='Feature Importance', rot=270)
    mng = pl.get_current_fig_manager()
    mng.window.showMaximized()
    pl.show()

model_standard = LogisticRegression(penalty='l1', C=100, solver='saga', max_iter=100)
model_standard.fit(X, y)

make_plot(model)
make_plot(model_standard)
"""
