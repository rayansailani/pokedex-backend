import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import joblib

app = Flask(__name__)

# Load the model and the label encoder
model = load_model('fine_tuned_model.keras')
le = joblib.load('label_encoder.pkl')
labels_encoded = np.load('pokemon_labels.npy')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_pokemon(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions, axis=1)[0]
    pokemon_name = le.inverse_transform([predicted_label])[0]
    return pokemon_name

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)
        pokemon_name = predict_pokemon(img_path)
        os.remove(img_path)
        return jsonify({'closest_pokemon': pokemon_name})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000, debug=True)
