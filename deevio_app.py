from PIL import Image
import numpy as np
import flask
import io
import sys
import pickle
import os

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
    sys.path.append(os.path.abspath("./model"))
    print(">>> Loading model")
    filename = './model/supervised_GB_model.pkl'
    model = pickle.load(open(filename, 'rb'))
    return model

def prepare_image(original_image, width=None, height=None):
    """Requires original image to be a numpy array and returns the scaled image as a numpy array.
    """
    original_image = np.array(original_image)

    h, w = original_image.shape
    print('The original image size is {wide} wide x {height} '
          'high'.format(wide=w, height=h))
 
    if width and height:
        max_size = (width, height)
    elif width:
        max_size = (width, h)
    elif height:
        max_size = (w, height)
    else:
        # No width or height specified
        raise RuntimeError('Width or height required!')
 
    original_image = Image.fromarray(original_image)
    
    #original_image.save('./test.jpeg')
    original_image.thumbnail(max_size, Image.ANTIALIAS)
    scaled_image = np.array(original_image)

    height, width = scaled_image.shape
    print('The scaled image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))
    
    scaled_image = scaled_image.reshape(1, height*width)
    return scaled_image

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    model = load_model()
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
     
            # preprocess the image and prepare it for classification
            image = prepare_image(image, width=1000)
            
            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            
            data["predictions"] = int(preds[0])

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()
