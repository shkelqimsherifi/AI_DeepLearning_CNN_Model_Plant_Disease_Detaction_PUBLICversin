# AI_DeepLearning_CNN_Model_Plant_Disease_Detaction

## Models PD36-B Architecture

<img width="6552" height="4153" alt="01_enhanced_architecture_diagram" src="https://github.com/user-attachments/assets/5bcc9354-c572-4c17-a3cc-0fc1742f9c5c" />

### Models PD36-B Architecture

![Model1ArchitectureStruct](https://github.com/user-attachments/assets/a8cd521f-db22-44af-882e-88a1724478b0)

### Models PD36-B Architecture

![Model1Architecture](https://github.com/user-attachments/assets/93ddee2a-d36c-4792-a3c5-e9d6476086b3)

### 

<img width="4727" height="4116" alt="03_parameter_analysis" src="https://github.com/user-attachments/assets/6de31142-d25f-4c7d-a049-90b143390432" />

<img width="4719" height="3534" alt="07_training_configuration" src="https://github.com/user-attachments/assets/b3080693-d84b-4280-9a75-52592f453ac8" />

<img width="2955" height="2348" alt="03_memory_usage_breakdown" src="https://github.com/user-attachments/assets/9ed705e9-817c-46d9-9a14-504dfcd4605a" />

### Model traning report

#### Check the detailed report of the model training history over 10 epochs.

report: [training_history_raw.csv](./training_history_raw.csv)


## Dataset

### Importing Dataset

```python

training_set = tf.keras.utils.image_dataset_from_directory(
    '/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    '/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

testing_set = tf.keras.utils.image_dataset_from_directory(
    '/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/test',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

```

<img width="6466" height="2356" alt="01_category_dis_Datase" src="https://github.com/user-attachments/assets/a2bd63e2-1174-47a7-8c10-ef28c8eec3d7" />

<img width="4589" height="2957" alt="01_category_distribution_heatmap" src="https://github.com/user-attachments/assets/41fef2ae-f68f-44e9-8654-ab9a10e513e6" />

<img width="3882" height="2350" alt="06_category_balance_analysis" src="https://github.com/user-attachments/assets/9434e6e4-5450-47f9-8073-0179ff03eb48" />

<img width="1596" height="590" alt="dataset" src="https://github.com/user-attachments/assets/e62e4dc1-614d-4f0a-9767-39a0d748121b" />


## Models PD36-B Confusion Matrix

<img width="4072" height="3829" alt="ConfusionMatrixx" src="https://github.com/user-attachments/assets/f9f01666-0bc2-4274-a5bc-83b600f98e0b" />

<img width="1200" height="1200" alt="PrecisionRecalF1Score" src="https://github.com/user-attachments/assets/f72a8698-ba9f-43f6-9659-4a600aecbe78" />

<img width="6034" height="1883" alt="classification_metrics" src="https://github.com/user-attachments/assets/e613f455-0882-48ea-aee1-c8e5c39e4f5e" />

### Full classification 

report: [Classification.csv](./Classification.csv)

### Summary

- Accuracy: **0.9657**
- Macro Avg F1: **0.9657**
- Weighted Avg F1: **0.9657**

## App UI

![Windows_App_UI_Qt](https://github.com/user-attachments/assets/fdd647bc-251f-47e5-9ab5-e957f9dbaf98)

![Screenshot 2026-01-26 010715](https://github.com/user-attachments/assets/0efce226-0140-4232-bfca-89d27f700b46)


### Models link:
1. PD36-A https://drive.google.com/file/d/1AUIx6B43zLuamtWcA5GRrJcpw35PTAMg/view?usp=sharing
2. PD36-B https://drive.google.com/file/d/1gBR4Ycu2qeKm0VDCaNaSh7IUyQlqd14G/view?usp=sharing

# Environment

1. Install NVidia Driver
2. Install Conda
3. Create Environment
``` conda create -n PlantDeseaseDetectionCNNModel python==3.10 ```
4. To activate this environment, use
``` conda activate PlantDeseaseDetectionCNNModel ```
 To deactivate an active environment, use
``` conda deactivate  ```
5. Check environment list
``` conda env list ```
6. GPU setup
``` conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 ```
7. Install TensorFlow
Update pip first
``` pip install --upgrade pip ```
or
``` python.exe -m pip install --upgrade pip ```
Important Anything above 2.10 is not supported on the GPU on #Windows native
``` pip install "tensorflow<2.11 ```
8. Install the requirements from requirement.txt on Created conda environment
``` pip install -r requirement.txt ```
9. Install jupyter lab
``` pip install jupyterlab ```
10. Dataset Kaggle
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download
11.  Tensorflow and Keras is used to train this model
Tensorflow - https://www.tensorflow.org/
Keras - https://keras.io/

### Requirements:

tensorflow==2.10.0
scikit-learn==1.3.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.13.0
pandas==2.1.0
streamlit
librosa==0.10.1
PySide6
opencv-python
numpy<2.0.0

## Model Architecture

```python

from tensorflow.keras.layers import Dense,Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3), padding='same',activation='relu',input_shape=(128,128,3)))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=64,kernel_size=(3,3), padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=128,kernel_size=(3,3), padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=256,kernel_size=(3,3), padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=512,kernel_size=(3,3), padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(units=1536,activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(units=38, activation='softmax'))  #38 is number of classes


metrics_list=['accuracy',
              tf.keras.metrics.F1Score(),
              tf.keras.metrics.TruePositives(),
              tf.keras.metrics.TrueNegatives(),
              tf.keras.metrics.FalsePositives(),
              tf.keras.metrics.FalseNegatives()
              ]

from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=metrics_list)

model.summary()

```

## Build and test the PD36-B

```python
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

```

```python

import sys
import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QLabel)
from PySide6.QtGui import QPixmap, QImage, QFont, QColor
from PySide6.QtCore import Qt, QSize
import cv2

# Fluent Widgets
from qfluentwidgets import (PushButton, PrimaryPushButton, TitleLabel, BodyLabel, 
                            CaptionLabel, MessageBox, MSFluentWindow, Theme, 
                            setTheme, ThemeColor, setThemeColor, CardWidget,
                            ImageLabel)
from qfluentwidgets import FluentIcon as FIF

from model_handler import ModelHandler
from treatments_en import treatments_en
from treatments_sq import treatments_sq

UI_LABELS = {
    "en": {
        "title": "Plant Disease Recognition",
        "load_model": "Load Model",
        "select_image": "Select Image",
        "predict": "Predict",
        "original_image": "Original Image",
        "prediction_result": "Prediction Result",
        "no_image": "No Image Selected",
        "status_ready": "Status: Ready",
        "status_loading": "Status: Model loaded from {filename}",
        "status_selected": "Status: Image selected: {filename}",
        "status_predicting": "Status: Predicting...",
        "status_complete": "Status: Prediction Complete",
        "status_error": "Status: Error during prediction.",
        "lang_btn": "Shqip",
        "treatment_title": "Treatment Information",
        "error_title": "Error"
    },
    "sq": {
        "title": "Identifikimi i Sëmundjeve",
        "load_model": "Ngarko Modelin",
        "select_image": "Zgjidh Imazhin",
        "predict": "Parashiko",
        "original_image": "Imazhi Origjinal",
        "prediction_result": "Rezultati",
        "no_image": "Asnjë imazh",
        "status_ready": "Statusi: Gati",
        "status_loading": "Statusi: Modeli u ngarkua nga {filename}",
        "status_selected": "Statusi: Imazhi u zgjodh: {filename}",
        "status_predicting": "Statusi: Duke parashikuar...",
        "status_complete": "Statusi: Përfundoi",
        "status_error": "Statusi: Gabim gjatë parashikimit.",
        "lang_btn": "English",
        "treatment_title": "Informacioni i Trajtimit",
        "error_title": "Gabim"
    }
}

class MainWindow(MSFluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plant Disease Detection")
        self.setMinimumSize(1000, 750)
        
        # Set Theme
        setTheme(Theme.LIGHT)
        setThemeColor("#1a73e8")

        self.model_handler = ModelHandler()
        self.selected_image_path = None
        self.current_lang = "en"

        self.init_ui()

    def init_ui(self):
        # Create a central widget for the content
        self.container = QWidget()
        self.container.setObjectName("homeInterface")
        self.main_layout = QVBoxLayout(self.container)
        self.main_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.setSpacing(25)

        # Header
        self.header_label = TitleLabel(UI_LABELS[self.current_lang]["title"])
        self.header_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.header_label)

        # Controls Layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        
        self.load_model_btn = PushButton(FIF.FOLDER, UI_LABELS[self.current_lang]["load_model"])
        self.load_model_btn.setFixedWidth(160)
        self.load_model_btn.clicked.connect(self.on_load_model)

        self.select_img_btn = PushButton(FIF.PHOTO, UI_LABELS[self.current_lang]["select_image"])
        self.select_img_btn.setFixedWidth(160)
        self.select_img_btn.clicked.connect(self.on_select_image)

        self.predict_btn = PrimaryPushButton(FIF.PLAY, UI_LABELS[self.current_lang]["predict"])
        self.predict_btn.setFixedWidth(160)
        self.predict_btn.clicked.connect(self.on_predict)
        self.predict_btn.setEnabled(False)

        self.lang_btn = PushButton(FIF.LANGUAGE, UI_LABELS[self.current_lang]["lang_btn"])
        self.lang_btn.setFixedWidth(130)
        self.lang_btn.clicked.connect(self.toggle_language)

        controls_layout.addStretch(1)
        controls_layout.addWidget(self.load_model_btn)
        controls_layout.addWidget(self.select_img_btn)
        controls_layout.addWidget(self.predict_btn)
        controls_layout.addWidget(self.lang_btn)
        controls_layout.addStretch(1)
        self.main_layout.addLayout(controls_layout)

        # Content Area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left Card: Image Selection
        self.original_card = CardWidget()
        left_layout = QVBoxLayout(self.original_card)
        left_layout.setContentsMargins(15, 15, 15, 15)
        self.original_title = BodyLabel(UI_LABELS[self.current_lang]["original_image"])
        self.original_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.original_title)

        self.original_img_view = ImageLabel()
        self.original_img_view.setFixedSize(450, 450)
        self.original_img_view.setBorderRadius(8, 8, 8, 8)
        self.original_img_view.setText(UI_LABELS[self.current_lang]["no_image"])
        self.original_img_view.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.original_img_view)
        content_layout.addWidget(self.original_card)

        # Right Card: Prediction Results
        self.result_card = CardWidget()
        right_layout = QVBoxLayout(self.result_card)
        right_layout.setContentsMargins(15, 15, 15, 15)
        self.result_title = BodyLabel(UI_LABELS[self.current_lang]["prediction_result"])
        self.result_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.result_title)

        self.predicted_img_view = ImageLabel()
        self.predicted_img_view.setFixedSize(450, 450)
        self.predicted_img_view.setBorderRadius(8, 8, 8, 8)
        self.predicted_img_view.setText(UI_LABELS[self.current_lang]["no_image"])
        self.predicted_img_view.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.predicted_img_view)

        self.result_label = BodyLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1a73e8;")
        right_layout.addWidget(self.result_label)
        
        content_layout.addWidget(self.result_card)
        self.main_layout.addLayout(content_layout)

        # Status Footer
        self.status_label = CaptionLabel(UI_LABELS[self.current_lang]["status_ready"])
        self.main_layout.addWidget(self.status_label)

        # Add central widget to FluentWindow
        self.addSubInterface(self.container, FIF.HOME, "Home")

    def toggle_language(self):
        self.current_lang = "sq" if self.current_lang == "en" else "en"
        self.update_ui_text()

    def update_ui_text(self):
        labels = UI_LABELS[self.current_lang]
        self.header_label.setText(labels["title"])
        self.load_model_btn.setText(labels["load_model"])
        self.select_img_btn.setText(labels["select_image"])
        self.predict_btn.setText(labels["predict"])
        self.lang_btn.setText(labels["lang_btn"])
        self.original_title.setText(labels["original_image"])
        self.result_title.setText(labels["prediction_result"])
        
        if not self.selected_image_path:
            self.original_img_view.setText(labels["no_image"])
            self.predicted_img_view.setText(labels["no_image"])

        # Update status label
        curr_status = self.status_label.text()
        if "Ready" in curr_status or "Gati" in curr_status:
            self.status_label.setText(labels["status_ready"])
        elif "Complete" in curr_status or "Përfundoi" in curr_status:
            self.status_label.setText(labels["status_complete"])

    def on_load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Keras Model (*.keras *.h5)")
        if file_path:
            success, message = self.model_handler.load_model(file_path)
            if success:
                self.status_label.setText(UI_LABELS[self.current_lang]["status_loading"].format(filename=os.path.basename(file_path)))
                if self.selected_image_path:
                    self.predict_btn.setEnabled(True)
            else:
                MessageBox("Error", f"Failed to load model: {message}", self).exec()

    def on_select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Images (*.png *.jpg *.jpeg *.JPG)")
        if file_path:
            self.selected_image_path = file_path
            self.original_img_view.setImage(file_path)
            self.status_label.setText(UI_LABELS[self.current_lang]["status_selected"].format(filename=os.path.basename(file_path)))
            
            if self.model_handler.model:
                self.predict_btn.setEnabled(True)

    def on_predict(self):
        if not self.selected_image_path:
            return

        self.status_label.setText(UI_LABELS[self.current_lang]["status_predicting"])
        result, error = self.model_handler.predict(self.selected_image_path)
        
        if error:
            MessageBox(UI_LABELS[self.current_lang]["error_title"], error, self).exec()
            self.status_label.setText(UI_LABELS[self.current_lang]["status_error"])
            return

        # Display result text
        label = result['label'].replace('___', ' - ')
        confidence = result['confidence'] * 100
        self.result_label.setText(f"{label}\n({confidence:.2f}%)")

        self.predicted_img_view.setImage(self.selected_image_path)
        self.status_label.setText(UI_LABELS[self.current_lang]["status_complete"])

        # Show Treatment Notification
        treatments = treatments_en if self.current_lang == "en" else treatments_sq
        original_label = result['label']
        treatment_text = treatments.get(original_label, "No treatment info available." if self.current_lang == "en" else "Nuk ka informacion për trajtimin.")
        
        # Modern Dialog for treatment
        title = UI_LABELS[self.current_lang]["treatment_title"]
        msg = MessageBox(title, treatment_text, self)
        msg.yesButton.setText("OK")
        msg.cancelButton.hide()
        msg.exec()

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


```

```python

import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


```

report: [treatments_en.py](./treatments_en.py)
report: [treatments_sq.py](./treatments_sq.py)

---

## Quick Start — Testing the Model Without the Application

> If you are **not comfortable running the full desktop application** or setting up the complete environment, you can use the provided script  
> **[Testing_Model2_Plant_Disease.ipynb](./Testing_Model2_Plant_Disease.ipynb)** to **quickly test the model and perform predictions** on plant leaf images.

### What does this script do?

The notebook provides a **simple, step-by-step** workflow:

1. **Loads the pre-trained model** (`trained_model2.keras`) — no training required.
2. **Displays the model architecture summary** so you can inspect the network.
3. **Reads a test image** of a plant leaf (supports `.jpg`, `.JPG`, `.png`).
4. **Visualizes the image** using Matplotlib.
5. **Predicts the disease class** and outputs the result.

### How to use it

1. **Open the notebook** in Jupyter Notebook.
2. **Upload the model file** `trained_model2.keras` to the same working directory (or adjust the path in the notebook).  
   You can download the model from the links provided in the [Models link](#models-link) section above.
3. **Upload a plant leaf image** you want to test. The notebook comes with several example images you can try:
   - `AppleCedarRust1.JPG`
   - `CornCommonRust2.JPG`
   - `PotatoHealthy1.JPG`
   - `TomatoYellowCurlVirus1.JPG`
   - `test_Yellow_LC.jpg`
4. **Run all cells** sequentially — the notebook will load the model, display your image, and print the predicted disease label.

### Why use this script?

| Full Desktop App | Testing Script (Notebook) |
|---|---|
| Requires Python environment, Conda, GPU drivers, PySide6, and multiple dependencies | Only requires Python with TensorFlow, NumPy, OpenCV, and Matplotlib |
| Designed for interactive, ongoing use | Designed for quick, one-off predictions |
| Provides a full GUI with treatment information | Provides a lightweight command-line/notebook experience |
| Best for researchers and developers | **Best for non-technical users** who just want to test the model |

> **Tip:** Google Colab is the easiest way to run the notebook — it provides a free cloud environment with all dependencies pre-installed, so you don't need to install anything on your machine.

---

### Read me update is coming soon!

