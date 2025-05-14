import os
import cv2
import numpy as np
import base64
from keras.models import model_from_json
from flask import Flask, request, jsonify

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the model
with open("signlanguagedetectionmodel48x481.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("signlanguagedetectionmodel48x481.h5")

# Assuming 'label' contains the list of labels for the output classes
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  # Example, replace with actual labels

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming the image is sent as a base64 string
        image_data = request.json['image']
        
        # Decode the base64 string
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))
        features = extract_features(image)

        # Make prediction
        pred = model.predict(features)
        prediction_label = label[pred.argmax()]
        accuracy = "{:.2f}".format(np.max(pred) * 100)
        
        return jsonify({'label': prediction_label, 'accuracy': accuracy})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
