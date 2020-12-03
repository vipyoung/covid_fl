# This code is an adaption of:
# https://github.com/ahmedfgad/FederatedLearning.git

import numpy as np
import socket
import pickle
import time
import threading

#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages

from sklearn.linear_model import LogisticRegression

class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, buffer_size=1048576, recv_timeout=5):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data += data
                if data == b'': # Nothing received from the client.
                    received_data = b""
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute, return with status 0 to close the connection.
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0 # 0 means the connection is no longer active and it should be closed.
                elif str(data)[-2] == '.':
                    print("All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info, data_len=len(received_data)))
                    if len(received_data) > 0:
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1
                        except BaseException as e:
                            print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0
                else:
                    self.recv_start_time = time.time()
            except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def reply(self, received_message):
        global server_model, client_samples, client_models, client_scores, confusion_matrix, acc_means, acc_stds, i_fold, k_folds, i_iter, i_reg, max_iter, reg_strengths
        if (type(received_message) is dict):
            response_message = {'what2do':'done'}
            if (i_iter == max_iter): # or any other condition to test the convergence
                for client_name, n_samples in client_samples.items():
                    if client_info in client_scores:
                        confusion_matrix[:,:,i_fold] += client_scores[client_info]
                    client_scores = {}
                    client_samples = {}
                    client_models = {}
                if i_fold < k_folds:
                    acc = np.sum(np.diag(confusion_matrix[:,:,i_fold])) / np.sum(confusion_matrix[:,:,i_fold])
                    print('Fold {} done, accuracy = {}'.format(i_fold, acc))
                i_fold += 1
                if i_fold == k_folds:
                    acc = np.array([np.sum(np.diag(confusion_matrix[:,:,i])) / np.sum(confusion_matrix[:,:,i]) for i in range(k_folds)])
                    acc_means[i_reg] = np.mean(acc)
                    acc_stds[i_reg] = np.std(acc)
                    print('With reg. strength = {}, accuracy = {} +/- {}'.format(reg_strengths[i_reg], acc_means[i_reg], acc_stds[i_reg]))
                    i_reg += 1
                    if i_reg == len(reg_strengths):
                        print('DONE')
                        print('Accuracy means:',acc_means)
                        print('Accuracy stds:',acc_stds)
                        #pdf = PdfPages('regularization.pdf')
                        #f = plt.figure()
                        #ax = f.add_subplot(1,1,1)
                        #ax.errorbar(reg_strengths, acc_means, yerr = acc_stds)
                        #pdf.savefig()
                        #plt.close()
                        #pdf.close()
                        response_message['what2do'] = 'done'
                    else:
                        i_fold = 0
                if i_fold < k_folds:
                    i_iter = 0
                    response_message['what2do'] = 'minimize'
                    server_model = LogisticRegression(penalty = 'l1', C = reg_strengths[i_reg], solver='saga', max_iter=1, warm_start=True)
                    response_message['model'] = server_model
                    response_message['i_fold'] = i_fold
            else:
                if received_message['what2do'] == 'ready':
                    response_message['what2do'] = 'minimize'
                    response_message['i_fold'] = i_fold
                    response_message['model'] = server_model
                elif received_message['what2do'] == 'update':
                    client_models[self.client_info] = received_message['model']
                    client_scores[self.client_info] = received_message['confusion_matrix']
                    client_samples[self.client_info] = received_message['n_samples']
                    response_message['what2do'] = 'echo'
                    #time.sleep(1)
                elif received_message['what2do'] == 'reset':
                        response_message['what2do'] = 'split'
                        response_message['k_folds'] = k_folds
                        response_message['model'] = server_model
                elif received_message['what2do'] == 'echo':
                    if client_samples.get(self.client_info, 0) == 0: # it means that the server model is new for the client
                        response_message['what2do'] = 'minimize'
                        response_message['i_fold'] = i_fold
                        response_message['model'] = server_model
                    else:
                        response_message['what2do'] = 'echo'
                    #time.sleep(1)
                update_server_model()
            response_data = pickle.dumps(response_message)
            try:
                self.connection.sendall(response_data)
            except BaseException as e:
                print('Error Sending Data to the Client: {msg}.\n'.format(msg=e))
        else:
            print('A dictionary is expected to be received from the client but {d_type} received'.format(d_type=type(received_data)))

    def run(self):
        try:
            self.connection.sendall(pickle.dumps({'what2do':'reset'}))
        except BaseException as e:
            print("Error Sending Data to the Client: {msg}.\n".format(msg=e))
        print("Running a Thread for the Connection with {client_info}".format(client_info=self.client_info))
        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            print('Waiting to Receive Data Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT'.format(year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec))
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print("Connection Closed with {client_info} either due to inactivity for {recv_timeout} seconds or due to an error.".format(client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
                break
            self.reply(received_data)

def update_server_model():
    global server_model, client_samples, client_models, i_iter, i_fold, i_reg, min_updated_clients
    print('SERVER MODEL:',server_model)
    print('CLIENT_SAMPLES:',client_samples)
    print('CLIENT_MODELS:',client_models)
    print('I_ITER:',i_iter,'I_FOLD:',i_fold,'I_REG:',i_reg)
    n_updates_from_clients = np.sum([n_samples > 0 for client_name, n_samples in client_samples.items()])
    shape_coefs = [model.coef_.shape for model in client_models.values() if hasattr(model, 'coef_')]
    if len(set(shape_coefs)) > 1:
        raise ValueError('Error: Model sizes do not agree')
    if n_updates_from_clients >= min_updated_clients:
        n_total_samples = sum(client_samples.values())
        server_model.coef_ = np.zeros(shape_coefs[0])
        server_model.intercept_ = 0
        for client_name, n_samples in client_samples.items():
            server_model.coef_ +=  client_models[client_name].coef_ * n_samples / n_total_samples
            server_model.intercept_ +=  client_models[client_name].intercept_ * n_samples / n_total_samples
            client_samples[client_name] = 0
        i_iter += 1

#----- INITIALIZATION
reg_strengths = np.logspace(np.log10(1e-2), np.log10(1e3), 10) # array with the tested values for regularization strengths
k_folds = 4 # number of folds for cross-validation
max_iter = 10 # maximum number of minimization steps
i_reg = 0 # index in the array of regularization strenths
i_fold = 0 # index of the fold for K-folds cross-validation
i_iter = 0 # index of the minimization step
min_updated_clients = 1 # the server model is updated only after receiving at least this number of updates from the clients
confusion_matrix = np.zeros((2,2,k_folds)) # this one is used to store the confusion matrixes during K-folds cross-validation
client_samples = {} # this is used for the number of samples in the last client update
client_scores = {} # ... for the confusion matrix in the last client update
client_models = {} # ... the model in the last client update
acc_means = np.empty(len(reg_strengths)) # average accuracy over k-folds at different regularization strengths
acc_stds = np.empty(len(reg_strengths)) # standard deviation of the accuracy over k-folds at different regularization strengths
server_model = LogisticRegression(penalty = 'l1', C = reg_strengths[i_reg], solver='saga', max_iter=1, warm_start=True)
server_ip = 'localhost'
server_port = 10000
#----- END: INITIALIZATION

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
soc.bind((server_ip, server_port))
soc.listen(1)
print("Socket is Listening for Connections...")

while True:
    try:
        connection, client_info = soc.accept()
        print("New Connection from: {client_info}".format(client_info=client_info))
        socket_thread = SocketThread(connection=connection, client_info=client_info, buffer_size=1048576, recv_timeout=10*60)
        socket_thread.start()
    except:
        soc.close()
        print("(Timeout) Socket Closed Because no Connections Received\n")
        break
