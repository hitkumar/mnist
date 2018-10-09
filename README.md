Training a neural network on MNIST Dataset.

There are three main files:

1. training.py -  Trains a neural network model on MNIST dataset and saves the model to disk.
Simply execute the python file to train the network - python training.py.

2. inference_server.py - Implements a REST API which accepts a raw image as input and returns the predicted number along with its probability as output. 

To start the server -  python inference_server.py

After the server starts, you can execute curl commands using POST to predict the number for an image:
curl -X POST -F image=@<image_file_path> 'http://localhost:5000/predict'

Example curl command - curl -X POST -F image=@img_1.jpg 'http://localhost:5000/predict'

Output - {"predicted_number":"0","probability":"0.9997091","success":true}

3. utils.py - Contains util functions useful in building training and inference pipelines like preprocessing data.

