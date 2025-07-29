import sys
import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
import traceback

# Add source directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'data_pipeline')))

# Import local utilities
from preprocess1 import preprocess_image_for_inference
from .ood_utils import is_in_distribution

# Load model and template
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "ensemble_model.keras"))
TEMPLATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CheXpert-v1.0', 'Template.jpg'))

model = load_model(MODEL_PATH)
TEMPLATE = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

if TEMPLATE is None or TEMPLATE.shape != (224, 224):
    raise ValueError(f"Template not found or incorrect shape: {TEMPLATE_PATH}")

# Flask config
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                image_array = preprocess_image_for_inference(filepath, TEMPLATE)
                print(f"[DEBUG] Preprocessed image shape: {image_array.shape}")
                if image_array is None or image_array.shape != (224, 224, 3):
                    raise ValueError(f"Invalid preprocessed image shape: {image_array.shape}")
            except Exception as e:
                traceback.print_exc()
                return render_template("index.html", result=f"Preprocessing error: {e}")

            if not is_in_distribution(image_array):
                return render_template("index.html", result="Image rejected. Please upload a clear and centered AP chest X-ray.")

            # Expand dims for batch and run inference
            image_batch = np.expand_dims(image_array, axis=0)
            preds = model.predict(image_batch, verbose=0)[0]

            labels = [
                'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                'Fracture', 'Support Devices'
            ]
            prediction = {label: float(prob) for label, prob in zip(labels, preds)}

            return render_template("index.html", result="Scan accepted.", predictions=prediction)

        return render_template("index.html", result="Invalid file type.")
    
    return render_template("index.html")