
# coding: utf-8

# In[ ]:


from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from keras.models import load_model


# In[ ]:


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None


# In[ ]:


def get_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = load_model('mnist_model.h5')


# In[ ]:


get_model()


# In[ ]:


# input image dimensions
img_rows, img_cols = 28, 28

def prepare_image(image):
    image = img_to_array(image)
    image = image.reshape(1, img_rows, img_cols, 1)
    image = image.astype('float32') / 255
    return image


# In[ ]:


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image)
            print (image.shape)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            model = load_model('mnist_model.h5')
            model._make_predict_function()
            preds = model.predict(image)
            print (preds)
            y_classes = preds.argmax(axis=-1)
            print (y_classes)		
            #results = imagenet_utils.decode_predictions(preds)
            #data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            #for (imagenetID, label, prob) in results[0]:
             #   r = {"label": label, "probability": float(prob)}
              #  data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# In[ ]:


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    # get_model()
    model._make_predict_function()
    app.run()

