import os
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Initialize Flask app
app = Flask(__name__)

# Upload folder configuration
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = load_model("mobilenetv2_plant_disease_final.keras")

# Load correct class mapping
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> class name
class_names = {v: k for k, v in class_indices.items()}

# Basic treatment suggestions dictionary
treatments = {
    "Late blight": "Apply copper-based fungicide and remove infected leaves.",
    "Early blight": "Use fungicide spray and maintain crop rotation.",
    "Bacterial spot": "Avoid overhead watering and remove infected leaves.",
    "Powdery mildew": "Apply sulfur-based fungicide and improve air circulation.",
    "Leaf Mold": "Reduce humidity and use resistant varieties.",
    "healthy": "No disease detected. Maintain proper plant care."
}

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    disease_raw = class_names[predicted_index]

    # Format clean name
    disease_name = disease_raw.replace("___", " - ").replace("_", " ")

    # Extract short disease name for treatment lookup
    disease_keyword = disease_name.split("-")[-1].strip()

    treatment = treatments.get(
        disease_keyword,
        "Consult an agricultural expert for proper treatment."
    )

    return disease_name, round(confidence * 100, 2), treatment


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    disease, confidence, treatment = predict_disease(filepath)

    return render_template(
        "result.html",
        prediction=disease,
        confidence=confidence,
        treatment=treatment,
        image_path=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)