import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import joblib
import numpy as np
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)

allowed_origins = [
    'http://localhost:3000', # for testing
    # add your applications frontend endpoint here.
    'https://pokedex-frontend-rho.vercel.app/'
]

CORS(app)  # This will enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load the fine-tuned model
model = load_model('fine_tuned_model.keras')

# Load the label encoder and labels
le = joblib.load('label_encoder.pkl')
labels_encoded = np.load('pokemon_labels.npy')

def find_closest_pokemon(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    img_features = model.predict(img_array)
    distances = np.linalg.norm(labels_encoded - img_features, axis=1)
    closest_pokemon_index = np.argmin(distances)
    closest_pokemon_label = le.inverse_transform([closest_pokemon_index])[0]
    return closest_pokemon_label

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    img = Image.open(file_path)
    closest_pokemon = find_closest_pokemon(img)
    
    os.remove(file_path)  # Clean up the saved file
    
    return jsonify({'closest_pokemon': closest_pokemon})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)