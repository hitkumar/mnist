''' Implements a REST API which accepts a raw image as input and returns the predicted number along with its probability as output. 
Using Flask, a Python web framework for building the API.

USAGE
Start the server:
    python run_keras_server.py
Submit a request via CURL: 
    curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
'''

from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from keras.models import load_model
from utils import prepare_images

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

@app.route("/predict", methods=["POST"])
def predict():

    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image, _ = prepare_images(image)

            # classify the input image and return the predicted number along with its probability.
            preds = model.predict(image)
            # indicate that the request was a success
            data["success"] = True
            data["predicted_number"] = str(preds.argmax(axis=-1)[0])
            data["probability"] = str(preds.max(axis=-1)[0])

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    model = load_model('mnist_model.h5')
    app.run()
