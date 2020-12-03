# Federated Learning for WES/WGS working group

Code for federated learning of Machine Learning models over the consurtium. A description of the algoritms can be found here:

https://docs.google.com/document/d/1SETsJs77z-32fHzovNEPhMOapK959w-u8PVynYCh8Fg/edit?usp=sharing

The libraries *socket* and *threading* are used to manage communications between the server and the client

The *sklearn* library is used for the ML model


* server.py: server code. It creates the sockets for communicating with the clients and it manages the federated learning process

* prepare_inputs.sh: It creates input files for the client

* client.py: client code. When requested by the server it performs a step of minimization of the loss function, and it sends back the updated model

For testing the code locally:

* 1) run prepare_inputs.sh to create the necessary input files

* 2) run the server with: python server.py

* 3) run as many clients as you want with: python client.py

The variable *min_updated_clients* in server.py defines how many clients needs to be updated before the server model is updated

If min_updated_clients is set to 1 and only one client is executed and the code reproduces a local (not-federated) learning of the model

In client.py the variable *X* is the matrix of input features with samples along the row, and *y* is the array of output classes



