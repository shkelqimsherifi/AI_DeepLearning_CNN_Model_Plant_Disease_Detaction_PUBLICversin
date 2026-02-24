import numpy as np
import tensorflow as tf
import cv2

class ModelHandler:
    def __init__(self):
        self.model = None
        self.class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]

    def load_model(self, model_path):
        """Loads a Keras model from the given path."""
        try:
            self.model = tf.keras.models.load_model(model_path)
            return True, "Model loaded successfully."
        except Exception as e:
            return False, str(e)

    def predict(self, image_path):
        """Preprocesses the image and makes a prediction."""
        if self.model is None:
            return None, "Model not loaded."

        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (128, 128))
            input_arr = np.array([img_resized]) # Convert single image to a batch.

            # Prediction
            prediction = self.model.predict(input_arr)
            result_index = np.argmax(prediction)
            
            label = self.class_names[result_index]
            confidence = float(np.max(prediction))
            
            return {
                "label": label,
                "confidence": confidence,
                "index": int(result_index)
            }, None
        except Exception as e:
            return None, str(e)
