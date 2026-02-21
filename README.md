AgroDetect AI is a Machine learning–based plant disease classification system that detects diseases from leaf images. The model uses MobileNetV2 with transfer learning and is trained on the PlantVillage dataset containing 38 disease classes.

Users can upload a plant leaf image through a web interface, and the system predicts the disease along with a confidence score and treatment recommendation. The application is built using TensorFlow for model training and Flask for web deployment.

1.mobilenetv2_final.keras

The trained deep learning model file used for disease prediction.

2.class_indices.json

Stores the mapping of class names to numerical labels. Ensures correct disease prediction in the web app.

3.static/style.css

Contains all styling for the web interface including layout, colors, and interactive design.

4.templates/home.html

Landing page of the application.

5.templates/about.html

Provides information about the model and technology used.

6.templates/upload.html

Allows users to upload plant leaf images.

7.templates/result.html

Displays predicted disease, confidence score, and treatment suggestions.

8.app.py

The Flask backend application.
Handles:

  Loading the trained model
  
  Receiving uploaded images
  
  Running predictions
  
  Sending results to the frontend
