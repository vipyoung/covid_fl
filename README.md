# Federated Learning for WES/WGS working group

Code for federated learning of Machine Learning models over the consurtium. A description of the algoritms can be found here:

https://docs.google.com/document/d/1SETsJs77z-32fHzovNEPhMOapK959w-u8PVynYCh8Fg/edit?usp=sharing

The libraries *socket* and *threading* are used to manage communications between the server and the client

The *sklearn* library is used for the ML model

* server.py: server code
It creates the sockets for communicating with the clients and it manages the federated learning of the model

* client.py: client code
When requested by the server it performs a step of minimization of the loss function, and it sends back the updated model

